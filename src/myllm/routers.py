from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from langchain_openai import ChatOpenAI
import os
from langserve import RemoteRunnable

class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: SecretStr
    model_name: str
    def __init__(
            self,
            model_name: str,
            openai_api_base: str = "https://openrouter.ai/api/v1",
            **kwargs):
        openai_api_key = os.getenv("OPENROUTER_API_KEY")
        super().__init__(openai_api_base=openai_api_base,
                        openai_api_key=openai_api_key,
                        model_name=model_name,
                        **kwargs)