GENERAL_MODELS = (
    "Claude-Opus-4.8",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3",
    "DeepSeek-V4-Pro",
    "Kimi-K3",
    "Qwen3.8-Max",
    "GPT-5.6-Sol",
    "Qwen3.8-27B",
    "Qwen3-8B",
    "GPT-OSS-20B",
    "Llama-3.1-8B-Instruct",
)

GUARD_MODELS = (
    "Llama-Guard-3-8B",
    "Qwen3Guard-Gen-8B",
    "AgentDoG1.5-Qwen3.5-4B",
)

EVALUATED_MODELS = GENERAL_MODELS + GUARD_MODELS + ("ObligationGuard",)
