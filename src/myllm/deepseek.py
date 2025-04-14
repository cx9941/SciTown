from langchain_openai import ChatOpenAI

class ChatLocalVLLM(ChatOpenAI):
    def __init__(
        self,
        model_name: str = "deepseek-v3:671b",
        openai_api_base: str = "https://uni-api.cstcloud.cn/v1",  # 正确的 vllm 原生接口
        openai_api_key: str = "6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b",  # vllm 不校验 key
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            **kwargs
        )

llm = ChatLocalVLLM(
    temperature=0.3,
)

from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content="你好，请简单介绍一下你自己")])
print(response.content)