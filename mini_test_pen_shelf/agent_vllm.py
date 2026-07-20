"""
agent_vllm.py — 用 vLLM 加载 Qwen3-4B，作为 ALFWorld 决策 agent

只做一件事：给定「已经用 prompt 模板拼好的 observation 文本」，返回模型生成的
原始字符串（含 <think>...</think> 和 <action>...</action>）。

显存：Qwen3-4B bf16 约 8-9GB，配合 gpu_memory_utilization 控制，可在 3090(24G) 快速跑。
"""
import os

# 关键：vLLM v1 用多进程拉起 EngineCore。父进程在 fork 前已初始化 CUDA，
# fork 出的子进程无法再 init CUDA -> "Cannot re-initialize CUDA in forked subprocess"。
# 强制 worker 用 spawn 启动方式即可解决。必须在 import vllm 之前设好。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class VLLMAgent:
    def __init__(
        self,
        model_path=None,
        gpu_memory_utilization=0.55,
        max_model_len=8192,
        max_tokens=5120,   # Thinking 模型推理很长；给足预算避免在 </think> 之前被截断
        temperature=0.4,
        enable_thinking=True,
        seed=0,
        tensor_parallel_size=1,
        no_wait=False,     # NoWait: 抑制 "Wait/Hmm/Alternatively..." 回溯词。默认关闭，
                           # 需要时显式开启；完整响应仍统一受 max_tokens 限制。
        think_budget=3500, # 兼容旧命令行；单次生成不再切分 think/action 预算
        action_budget=256, # 兼容旧命令行；完整响应统一受 max_tokens 限制
        pipeline_parallel_size=1,
        max_num_seqs=None,
        enforce_eager=True,
        force_action_prefix=False,  # 兼容旧消融参数；单次生成时不再强制续写前缀
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        model_path = model_path or os.environ.get("MODEL_PATH")
        assert model_path, "请 export MODEL_PATH=/path/to/Qwen3-4B 或传入 model_path"

        print(f"[vLLM] 加载模型: {model_path}")
        print(f"[vLLM] gpu_mem_util={gpu_memory_utilization}, max_model_len={max_model_len}, "
              f"tensor_parallel_size={tensor_parallel_size}, "
              f"pipeline_parallel_size={pipeline_parallel_size}, "
              f"max_num_seqs={max_num_seqs}, enforce_eager={enforce_eager}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.max_model_len = int(max_model_len)
        # One request emits the complete strict protocol.  Reserve exactly the
        # advertised response budget; no generated thought is fed back as a
        # second prompt.
        self._prompt_token_limit = self.max_model_len - int(max_tokens)
        if self._prompt_token_limit <= 0:
            raise ValueError("max_model_len must leave room for max_tokens")
        self.context_guard_prompt_trims = 0
        self.context_guard_think_trims = 0
        self.context_guard_trimmed_tokens = 0
        self._context_guard_reported = False
        llm_kwargs = dict(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            # Eager is useful for tiny local smoke tests.  Production A800
            # runs use CUDA Graphs after warm-up to raise decode throughput;
            # this changes execution capture, not prompt/sampling semantics.
            enforce_eager=bool(enforce_eager),
            seed=seed,
        )
        # This rollout code never submits more than one environment batch at
        # once.  Bounding vLLM's scheduler avoids an unnecessary 1024-request
        # dummy sampler warm-up without changing any generated request.
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        self.llm = LLM(**llm_kwargs)
        # NoWait（论文版）：抑制"反思类"回溯词，迫使模型不反复回头、更快收敛到
        # <action>，缩短 thinking、减少 max_tokens 截断。
        # 论文做法 = 扫全词表，凡 decode 后命中 wait/hmm/alternatively/... 子串的
        # token 变体（Wait/ Wait/WAIT/wait,/.wait...）全部禁止。
        # vLLM V1 不支持 per-request logits_processors，改用 bad_words：把命中的 token
        # 单独 decode 成字符串传入，等价于"该 token 出现即禁"。
        bad_words = None
        if no_wait:
            bad_words = self._build_nowait_bad_words()
            print(f"[NoWait] 扫词表得到 {len(bad_words)} 个回溯词 token 变体")

        self.sampling = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            bad_words=bad_words,
            stop=["</action>"],
            include_stop_str_in_output=True,
        )
        # ``think_budget`` / ``action_budget`` remain accepted so old launchers
        # and checkpoints can be resumed, but the standard single-pass decoder
        # uses only max_tokens for the complete think+action response.
        self.max_tokens = max_tokens
        self.think_budget = think_budget
        self.action_budget = action_budget
        self._bad_words = bad_words
        self._temperature = temperature
        self.force_action_prefix = bool(force_action_prefix)
        if self.force_action_prefix:
            print("[vLLM] force_action_prefix is ignored by single-pass generation")
        self.enable_thinking = enable_thinking
        # Exact inference-token accounting from vLLM RequestOutput objects.
        # Each active environment decision now contributes exactly one prompt
        # and one complete sampled response.
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        # Per-request breakdown from the most recent generate() call, aligned
        # index-for-index with that call's prompts/request_outputs. Callers
        # that batch many episode slots into one generate() (e.g.
        # act_batch_with_meta) use this to attribute exact tokens back to the
        # individual slot that owned each request, since the cumulative
        # counters above can't be un-summed after the fact.
        self.last_batch_request_tokens = []

    def _single_sampling(self, *, temperature=None, seed=None):
        """Build request-local single-pass sampling parameters when overridden.

        Held-out validation uses a fixed request seed and lower temperature.
        Keeping it request-local prevents validation decoding from consuming the
        rollout RNG stream and changing subsequent training rollouts.
        """
        if temperature is None and seed is None:
            return self.sampling
        from vllm import SamplingParams
        value = self._temperature if temperature is None else float(temperature)
        request_seed = None if seed is None else int(seed)
        return SamplingParams(
            temperature=value,
            top_p=0.95,
            max_tokens=self.max_tokens,
            bad_words=self._bad_words,
            seed=request_seed,
            stop=["</action>"],
            include_stop_str_in_output=True,
        )

    def _record_token_usage(self, request_outputs):
        per_request = []
        for request_output in request_outputs:
            prompt_ids = getattr(request_output, "prompt_token_ids", None) or []
            p = len(prompt_ids)
            self.total_prompt_tokens += p
            outputs = getattr(request_output, "outputs", None) or []
            r = 0
            if outputs:
                token_ids = getattr(outputs[0], "token_ids", None) or []
                r = len(token_ids)
            self.total_response_tokens += r
            per_request.append({"prompt": p, "response": r, "total": p + r})
        self.last_batch_request_tokens = per_request

    def get_token_usage(self):
        """Return exact cumulative vLLM inference tokens for this agent."""
        prompt = int(self.total_prompt_tokens)
        response = int(self.total_response_tokens)
        return {"prompt": prompt, "response": response, "total": prompt + response}

    def close(self):
        """Shut down the vLLM EngineCore cleanly when a rollout worker exits."""
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        self.llm = None
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            # Interpreter shutdown may already have torn down torch modules.
            pass

    # 反思词子串（小写）。命中即视为回溯词——只放【不会误伤】的长词。
    # 注意 "wait"/"hmm" 不在此列：wait 是 Kuwait/await/WaitForSeconds 的子串，放子串组
    # 会误伤，故归入下面的严格相等组 _REFLECT_EXACT。
    _REFLECT_SUBSTRINGS = (
        "alternatively", "reconsider", "double-check", "double check", "recheck",
    )
    # 严格组：token 去掉首尾空白/标点后【完全等于】才禁，避免 Kuwait/await/button/
    # checkout 之类误伤。涵盖论文黑名单核心词。
    _REFLECT_EXACT = (
        "wait", "hmm", "but", "however", "check", "maybe", "verify", "actually",
    )
    # 人工白名单：含上述子串但【不该】禁的 token（防误伤）。例如某些含 "but" 的
    # 完整词已被 _REFLECT_EXACT 的严格相等规则挡掉，这里留作进一步人工排除的口子。
    _REFLECT_WHITELIST = ()

    def _build_nowait_bad_words(self):
        """扫整个 tokenizer 词表，收集所有"反思类"token 变体，返回 bad_words 字符串列表。
        - 子串命中（wait/hmm/alternatively/however/reconsider/double-check）：宽松，
          token 文本含该子串即收。
        - 短词（but/check/maybe/verify/actually）：严格，去掉首尾空白和标点后完全相等才收，
          避免 button/checkout 之类误伤。"""
        vocab = self.tokenizer.get_vocab()  # {token_str: id}
        bad = set()
        import re as _re
        for tid in range(len(vocab)):
            try:
                s = self.tokenizer.decode([tid])
            except Exception:
                continue
            low = s.lower()
            stripped = _re.sub(r"^[\s\W]+|[\s\W]+$", "", low)  # 去首尾空白/标点
            hit = False
            if any(sub in low for sub in self._REFLECT_SUBSTRINGS):
                hit = True
            elif stripped in self._REFLECT_EXACT:
                hit = True
            if hit and stripped and stripped not in self._REFLECT_WHITELIST:
                bad.add(s)
        return sorted(bad)

    def _build_prompt(self, obs_text):
        """单条 user message -> chat template 字符串（与项目 rollout_loop 一致）。"""
        chat = [{"role": "user", "content": obs_text}]
        # Qwen3 支持 enable_thinking 开关
        try:
            prompt = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # 某些 tokenizer 不接受 enable_thinking 参数
            prompt = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
        return self._fit_initial_prompt(prompt)

    def _token_ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode_ids(self, token_ids):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _trim_middle_tokens(self, token_ids, token_limit):
        """Keep both task/header and latest observation/action suffix."""
        if len(token_ids) <= token_limit:
            return token_ids, 0
        marker = self._token_ids("\n[Earlier prompt context omitted for context limit.]\n")
        if token_limit <= len(marker) + 2:
            kept = token_ids[-token_limit:]
        else:
            usable = token_limit - len(marker)
            head = usable // 2
            tail = usable - head
            kept = token_ids[:head] + marker + token_ids[-tail:]
        return kept, len(token_ids) - len(kept)

    def _report_context_guard(self, kind, removed):
        if removed <= 0:
            return
        self.context_guard_trimmed_tokens += int(removed)
        if not self._context_guard_reported:
            print(
                f"[vLLM][context-guard] {kind}: trimmed {removed} tokens "
                f"to preserve max_model_len={self.max_model_len} and the response budget"
            )
            self._context_guard_reported = True

    def _fit_initial_prompt(self, prompt):
        """Enforce the input budget before single-pass generation starts."""
        token_ids = self._token_ids(prompt)
        kept, removed = self._trim_middle_tokens(token_ids, self._prompt_token_limit)
        if removed:
            self.context_guard_prompt_trims += 1
            self._report_context_guard("initial prompt", removed)
            return self._decode_ids(kept)
        return prompt

    @staticmethod
    def _restore_think(prompt, text):
        """Thinking 模型的 chat template 会把开头的 <think> 注入到 prompt 末尾，
        于是生成文本里只剩 </think>，缺了开标签。projection 要求成对的
        <think>...</think> 才判 valid，这里补回开标签，保证下游解析正确。"""
        if "<think>" not in text and prompt.rstrip().endswith("<think>"):
            return "<think>\n" + text
        return text

    def act(self, obs_text):
        """返回模型生成的原始文本（单条），已补回开头的 <think> 标签。"""
        text, _ = self.act_with_meta(obs_text)
        return text

    def act_with_meta(self, obs_text):
        """Single-pass generation of the complete strict protocol.

        ``forced`` remains as a compatibility diagnostic and is true only when
        vLLM exhausts ``max_tokens``.  Missing tags are not fabricated here;
        the environment projection records such output as invalid.
        """
        prompt = self._build_prompt(obs_text)
        outs = self.llm.generate([prompt], self.sampling, use_tqdm=False)
        self._record_token_usage(outs)
        completion = outs[0].outputs[0]
        forced = getattr(completion, "finish_reason", None) == "length"
        return self._restore_think(prompt, completion.text), forced

    def act_batch(self, obs_texts):
        prompts = [self._build_prompt(t) for t in obs_texts]
        outs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        self._record_token_usage(outs)
        return [
            self._restore_think(p, o.outputs[0].text)
            for p, o in zip(prompts, outs)
        ]

    def act_batch_with_meta(self, obs_texts, *, temperature=None, sampling_seed=None,
                            sampling_seeds=None):
        """Batch version of :meth:`act_with_meta`, using one vLLM request row
        per active environment decision.

        This preserves the per-step prompt / sampling policy while letting vLLM
        batch many ALFWorld environments together. Returns ``[(text, forced), …]``
        in the same order as ``obs_texts``.
        """
        prompts = [self._build_prompt(t) for t in obs_texts]
        if sampling_seed is not None and sampling_seeds is not None:
            raise ValueError("use sampling_seed or sampling_seeds, not both")
        if sampling_seeds is not None:
            if len(sampling_seeds) != len(prompts):
                raise ValueError("sampling_seeds must match the prompt batch length")
            sampling = [
                self._single_sampling(temperature=temperature, seed=seed)
                for seed in sampling_seeds
            ]
        else:
            sampling = self._single_sampling(
                temperature=temperature, seed=sampling_seed)
        outs = self.llm.generate(prompts, sampling, use_tqdm=False)
        self._record_token_usage(outs)
        return [
            (
                self._restore_think(prompt, out.outputs[0].text),
                getattr(out.outputs[0], "finish_reason", None) == "length",
            )
            for prompt, out in zip(prompts, outs)
        ]
