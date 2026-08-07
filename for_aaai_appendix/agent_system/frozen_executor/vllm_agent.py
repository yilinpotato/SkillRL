""

import os




os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class VLLMAgent:
    def __init__(
        self,
        model_path=None,
        gpu_memory_utilization=0.55,
        max_model_len=8192,
        max_tokens=5120,
        temperature=0.4,
        enable_thinking=True,
        seed=0,
        tensor_parallel_size=1,
        no_wait=False,

        think_budget=3500,
        action_budget=256,
        pipeline_parallel_size=1,
        max_num_seqs=None,
        enforce_eager=True,
        force_action_prefix=False,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        model_path = model_path or os.environ.get("MODEL_PATH")
        assert model_path, "Set MODEL_PATH or pass model_path"

        print(f"[vLLM] model={model_path}")
        print(f"[vLLM] gpu_mem_util={gpu_memory_utilization}, max_model_len={max_model_len}, tensor_parallel_size={tensor_parallel_size}, pipeline_parallel_size={pipeline_parallel_size}, max_num_seqs={max_num_seqs}, enforce_eager={enforce_eager}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.max_model_len = int(max_model_len)



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



            enforce_eager=bool(enforce_eager),
            seed=seed,
        )



        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        self.llm = LLM(**llm_kwargs)






        bad_words = None
        if no_wait:
            bad_words = self._build_nowait_bad_words()
            print(f"[NoWait] blocked_token_variants={len(bad_words)}")

        self.sampling = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            bad_words=bad_words,
            stop=["</action>"],
            include_stop_str_in_output=True,
        )



        self.max_tokens = max_tokens
        self.think_budget = think_budget
        self.action_budget = action_budget
        self._bad_words = bad_words
        self._temperature = temperature
        self.force_action_prefix = bool(force_action_prefix)
        if self.force_action_prefix:
            print("[vLLM] force_action_prefix is ignored by single-pass generation")
        self.enable_thinking = enable_thinking



        self.total_prompt_tokens = 0
        self.total_response_tokens = 0






        self.last_batch_request_tokens = []

    def _single_sampling(self, *, temperature=None, seed=None):
        ""
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
        ""
        prompt = int(self.total_prompt_tokens)
        response = int(self.total_response_tokens)
        return {"prompt": prompt, "response": response, "total": prompt + response}

    def get_context_guard_usage(self):
        ""
        return {
            "prompt_trims": int(getattr(self, "context_guard_prompt_trims", 0) or 0),
            "trimmed_tokens": int(getattr(self, "context_guard_trimmed_tokens", 0) or 0),
        }

    def close(self):
        ""
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

            pass




    _REFLECT_SUBSTRINGS = (
        "alternatively",
        "reconsider",
        "double-check",
        "double check",
        "recheck",
    )


    _REFLECT_EXACT = (
        "wait",
        "hmm",
        "but",
        "however",
        "check",
        "maybe",
        "verify",
        "actually",
    )


    _REFLECT_WHITELIST = ()

    def _build_nowait_bad_words(self):
        ""
        vocab = self.tokenizer.get_vocab()
        bad = set()
        import re as _re

        for tid in range(len(vocab)):
            try:
                s = self.tokenizer.decode([tid])
            except Exception:
                continue
            low = s.lower()
            stripped = _re.sub(r"^[\s\W]+|[\s\W]+$", "", low)
            hit = False
            if any(sub in low for sub in self._REFLECT_SUBSTRINGS):
                hit = True
            elif stripped in self._REFLECT_EXACT:
                hit = True
            if hit and stripped and stripped not in self._REFLECT_WHITELIST:
                bad.add(s)
        return sorted(bad)

    def _build_prompt(self, obs_text):
        ""
        chat = [{"role": "user", "content": obs_text}]

        try:
            prompt = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:

            prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
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
        ""
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
            print(f"[vLLM][context-guard] {kind}: trimmed {removed} tokens to preserve max_model_len={self.max_model_len} and the response budget")
            self._context_guard_reported = True

    def _fit_initial_prompt(self, prompt):
        ""
        token_ids = self._token_ids(prompt)
        kept, removed = self._trim_middle_tokens(token_ids, self._prompt_token_limit)
        if removed:
            self.context_guard_prompt_trims += 1
            self._report_context_guard("initial prompt", removed)
            return self._decode_ids(kept)
        return prompt

    @staticmethod
    def _restore_think(prompt, text):
        ""
        if "<think>" not in text and prompt.rstrip().endswith("<think>"):
            return "<think>\n" + text
        return text

    def act(self, obs_text):
        ""
        text, _ = self.act_with_meta(obs_text)
        return text

    def act_with_meta(self, obs_text):
        ""
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
        return [self._restore_think(p, o.outputs[0].text) for p, o in zip(prompts, outs)]

    def act_batch_with_meta(self, obs_texts, *, temperature=None, sampling_seed=None, sampling_seeds=None):
        ""
        prompts = [self._build_prompt(t) for t in obs_texts]
        if sampling_seed is not None and sampling_seeds is not None:
            raise ValueError("use sampling_seed or sampling_seeds, not both")
        if sampling_seeds is not None:
            if len(sampling_seeds) != len(prompts):
                raise ValueError("sampling_seeds must match the prompt batch length")
            sampling = [self._single_sampling(temperature=temperature, seed=seed) for seed in sampling_seeds]
        else:
            sampling = self._single_sampling(temperature=temperature, seed=sampling_seed)
        outs = self.llm.generate(prompts, sampling, use_tqdm=False)
        self._record_token_usage(outs)
        return [
            (
                self._restore_think(prompt, out.outputs[0].text),
                getattr(out.outputs[0], "finish_reason", None) == "length",
            )
            for prompt, out in zip(prompts, outs)
        ]
