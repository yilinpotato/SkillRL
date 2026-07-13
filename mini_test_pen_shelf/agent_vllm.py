"""
agent_vllm.py — 用 vLLM 加载 Qwen3-4B，作为 ALFWorld 决策 agent

只做一件事：给定「已经用 prompt 模板拼好的 observation 文本」，返回模型生成的
原始字符串（含 <think>...</think> 和 <action>...</action>）。

显存：Qwen3-4B bf16 约 8-9GB，配合 gpu_memory_utilization 控制，可在 3090(24G) 快速跑。
"""
import os
import re

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
                           # 需要时显式开启（budget forcing 已能控制思考长度）。
        think_budget=3500, # 思考预算：第一阶段生成上限。到此还没 </think> 就强制收尾出 action
        action_budget=256, # 第二阶段动作预算；WebShop 用 128，使 640+128 对齐 response=768
        pipeline_parallel_size=1,
        max_num_seqs=None,
        enforce_eager=True,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        model_path = model_path or os.environ.get("MODEL_PATH")
        assert model_path, "请 export MODEL_PATH=/path/to/Qwen3-4B 或传入 model_path"

        print(f"[vLLM] 加载模型: {model_path}")
        print(f"[vLLM] gpu_mem_util={gpu_memory_utilization}, max_model_len={max_model_len}, "
              f"tensor_parallel_size={tensor_parallel_size}, "
              f"pipeline_parallel_size={pipeline_parallel_size}, "
              f"max_num_seqs={max_num_seqs}, enforce_eager={enforce_eager}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.max_model_len = int(max_model_len)
        # A forced close appends ``</think>`` after stage 1.  Reserve its exact
        # token length (with a small floor for tokenizer variants) so a prompt
        # at the advertised input budget still leaves room for stage 2.
        self._forced_close_token_slack = max(
            4,
            len(self.tokenizer.encode("\n</think>\n", add_special_tokens=False)),
        )
        self._prompt_token_limit = self.max_model_len - max(
            int(max_tokens),
            int(think_budget) + int(action_budget) + self._forced_close_token_slack,
        )
        self._action_prompt_token_limit = self.max_model_len - int(action_budget)
        if self._prompt_token_limit <= 0 or self._action_prompt_token_limit <= 0:
            raise ValueError(
                "max_model_len must leave room for the configured thinking and action budgets")
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
        )
        # Budget forcing 用：思考阶段（上限 think_budget），以及强制收尾后的动作阶段。
        self.max_tokens = max_tokens
        self.think_budget = think_budget
        self._bad_words = bad_words
        self._temperature = temperature
        # 思考阶段：到 think_budget 截断；遇到 </think> 也自然停（用 stop）。
        self.think_sampling = SamplingParams(
            temperature=temperature, top_p=0.95,
            max_tokens=think_budget, bad_words=bad_words,
            stop=["</think>"], include_stop_str_in_output=True,
        )
        # 动作阶段：思考已被强制关闭，只需很短长度吐出 <action>...</action>。
        self.action_sampling = SamplingParams(
            temperature=temperature, top_p=0.95,
            max_tokens=action_budget, stop=["</action>"], include_stop_str_in_output=True,
        )
        self.enable_thinking = enable_thinking
        self.action_budget = action_budget
        # Exact inference-token accounting from vLLM RequestOutput objects.
        # Prompt tokens count every model invocation.  With two-stage budget
        # forcing this intentionally includes both the thinking request and the
        # action request (whose prompt contains the generated thinking), because
        # both consume accelerator compute.
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0

    def _record_token_usage(self, request_outputs):
        for request_output in request_outputs:
            prompt_ids = getattr(request_output, "prompt_token_ids", None) or []
            self.total_prompt_tokens += len(prompt_ids)
            outputs = getattr(request_output, "outputs", None) or []
            if outputs:
                token_ids = getattr(outputs[0], "token_ids", None) or []
                self.total_response_tokens += len(token_ids)

    def get_token_usage(self):
        """Return exact cumulative vLLM inference tokens for this agent."""
        prompt = int(self.total_prompt_tokens)
        response = int(self.total_response_tokens)
        return {"prompt": prompt, "response": response, "total": prompt + response}

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
                f"to preserve max_model_len={self.max_model_len} and the configured action budget"
            )
            self._context_guard_reported = True

    def _fit_initial_prompt(self, prompt):
        """Enforce the declared input budget before stage-1 thinking starts."""
        token_ids = self._token_ids(prompt)
        kept, removed = self._trim_middle_tokens(token_ids, self._prompt_token_limit)
        if removed:
            self.context_guard_prompt_trims += 1
            self._report_context_guard("initial prompt", removed)
            return self._decode_ids(kept)
        return prompt

    def _fit_action_prompt(self, prompt, think_part):
        """Last-resort token guard for the prompt fed to stage-2 action decoding.

        The normal path is already bounded by :meth:`_fit_initial_prompt`.
        This second check covers tokenizer/output edge cases without letting
        vLLM fail an entire rollout batch for one overlong sample.
        """
        suffix = think_part + "\n"
        prefix_ids = self._token_ids(prompt)
        suffix_ids = self._token_ids(suffix)
        overflow = len(prefix_ids) + len(suffix_ids) - self._action_prompt_token_limit
        if overflow <= 0:
            return prompt + suffix, think_part

        # Preserve the end of the thought, including ``</think>``, because it
        # contains the most recent conclusion and keeps the strict protocol
        # well formed after _restore_think adds a missing opening tag.
        if overflow < len(suffix_ids):
            suffix_ids = suffix_ids[overflow:]
            fitted_think = self._decode_ids(suffix_ids).rstrip()
            self.context_guard_think_trims += 1
            self._report_context_guard("action-context thought", overflow)
            return prompt + self._decode_ids(suffix_ids), fitted_think

        # This is only reachable if the input itself was unexpectedly longer
        # than the bounded stage-1 limit.  Keep a compact head/tail prompt and
        # the complete protocol suffix rather than crashing the whole batch.
        allowed_prefix = max(1, self._action_prompt_token_limit - len(suffix_ids))
        kept_prefix, removed = self._trim_middle_tokens(prefix_ids, allowed_prefix)
        self.context_guard_prompt_trims += 1
        self._report_context_guard("action-context prompt", removed)
        return self._decode_ids(kept_prefix) + suffix, think_part

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
        """Budget forcing 两阶段生成，返回 (原始文本, forced)。
        阶段1：生成思考，上限 think_budget，遇 </think> 自停。
          - 若模型在阶段1就已经自然写出 </think> 并跟上 <action>，直接用，不走阶段2
            （避免阶段2再生成一个 action 与之拼接成畸形双 action）。
        阶段2：仅当阶段1撞预算（无 </think>）时，硬接 </think> 再生成 <action>，上限很短。
        forced=True 表示思考被预算强制截断。"""
        prompt = self._build_prompt(obs_text)

        # 阶段 1：思考（stop=</think>，include_stop_str_in_output 使输出含 </think>）
        out1 = self.llm.generate([prompt], self.think_sampling, use_tqdm=False)
        self._record_token_usage(out1)
        comp1 = out1[0].outputs[0]
        think_text = comp1.text
        got_close = "</think>" in think_text

        if got_close:
            # 模型自然收尾。截到第一个 </think> 为止（丢弃其后可能的畸形内容）。
            think_part = think_text[:think_text.index("</think>") + len("</think>")]
        else:
            # 撞 think_budget：硬接 </think> 强制收尾
            think_part = think_text.rstrip() + "\n</think>"
        forced = not got_close

        # 阶段 2：续写动作（无论阶段1是否自然收尾都重新干净生成一次 action，
        # 保证 <action> 紧跟在 </think> 之后、格式规整，不与阶段1残留拼接）
        action_prompt, think_part = self._fit_action_prompt(prompt, think_part)
        out2 = self.llm.generate([action_prompt], self.action_sampling, use_tqdm=False)
        self._record_token_usage(out2)
        action_text = out2[0].outputs[0].text.strip()
        # 只保留 action_text 里第一个 <action>...</action>，去掉多余内容
        ma = re.search(r"<action>.*?</action>", action_text, re.DOTALL | re.IGNORECASE)
        if ma:
            action_text = ma.group(0)

        full = think_part + "\n" + action_text
        return self._restore_think(prompt, full), forced

    def act_batch(self, obs_texts):
        prompts = [self._build_prompt(t) for t in obs_texts]
        outs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        self._record_token_usage(outs)
        return [
            self._restore_think(p, o.outputs[0].text)
            for p, o in zip(prompts, outs)
        ]

    def act_batch_with_meta(self, obs_texts):
        """Batch version of :meth:`act_with_meta` with the same two-stage
        budget-forcing logic.

        This preserves the per-step prompt / sampling policy while letting vLLM
        batch many ALFWorld environments together. Returns ``[(text, forced), …]``
        in the same order as ``obs_texts``.
        """
        prompts = [self._build_prompt(t) for t in obs_texts]

        outs1 = self.llm.generate(prompts, self.think_sampling, use_tqdm=False)
        self._record_token_usage(outs1)
        think_parts = []
        forced_flags = []
        for out in outs1:
            think_text = out.outputs[0].text
            got_close = "</think>" in think_text
            if got_close:
                think_part = think_text[:think_text.index("</think>") + len("</think>")]
            else:
                think_part = think_text.rstrip() + "\n</think>"
            think_parts.append(think_part)
            forced_flags.append(not got_close)

        fitted = [self._fit_action_prompt(p, t) for p, t in zip(prompts, think_parts)]
        action_prompts = [item[0] for item in fitted]
        think_parts = [item[1] for item in fitted]
        outs2 = self.llm.generate(action_prompts, self.action_sampling, use_tqdm=False)
        self._record_token_usage(outs2)

        results = []
        for prompt, think_part, forced, out in zip(prompts, think_parts, forced_flags, outs2):
            action_text = out.outputs[0].text.strip()
            ma = re.search(r"<action>.*?</action>", action_text, re.DOTALL | re.IGNORECASE)
            if ma:
                action_text = ma.group(0)
            full = think_part + "\n" + action_text
            results.append((self._restore_think(prompt, full), forced))
        return results
