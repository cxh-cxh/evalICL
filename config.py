class VLMConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


configs = {
    "qwen3-vl-plus": VLMConfig(
        model="qwen3-vl-plus",
        api_key="sk-5717b623e2574041b2be8879f7d44220",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": True, "thinking_budget": 81920},
    ),
    "gpt-4o": VLMConfig(
        model="gpt-4o",
        api_key="",
        base_url="https://api.chatanywhere.tech/v1",
        extra_body=None,
    ),
}
