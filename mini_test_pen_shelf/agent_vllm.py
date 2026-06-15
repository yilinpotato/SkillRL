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
        max_tokens=1024,
        temperature=0.4,
        enable_thinking=True,
        seed=0,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        model_path = model_path or os.environ.get("MODEL_PATH")
        assert model_path, "请 export MODEL_PATH=/path/to/Qwen3-4B 或传入 model_path"

        print(f"[vLLM] 加载模型: {model_path}")
        print(f"[vLLM] gpu_mem_util={gpu_memory_utilization}, max_model_len={max_model_len}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,   # 单任务调试，跳过 CUDA graph 编译省启动时间
            seed=seed,
        )
        self.sampling = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        self.enable_thinking = enable_thinking

    def _build_prompt(self, obs_text):
        """单条 user message -> chat template 字符串（与项目 rollout_loop 一致）。"""
        chat = [{"role": "user", "content": obs_text}]
        kwargs = {}
        # Qwen3 支持 enable_thinking 开关
        try:
            return self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # 某些 tokenizer 不接受 enable_thinking 参数
            return self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )

    def act(self, obs_text):
        """返回模型生成的原始文本（单条）。"""
        prompt = self._build_prompt(obs_text)
        out = self.llm.generate([prompt], self.sampling, use_tqdm=False)
        return out[0].outputs[0].text

    def act_batch(self, obs_texts):
        prompts = [self._build_prompt(t) for t in obs_texts]
        outs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outs]
