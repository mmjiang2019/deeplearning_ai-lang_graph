from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from lang_graph_project.config import open_ai as open_ai_config

def get_base_url() -> str:
    return open_ai_config.BASE_URL

def get_api_key() -> str:
    return open_ai_config.API_KEY

def create_model(model: str, temperature: float = 0.0, model_provider: str = "openai"):
    chat_model = init_chat_model(
        model=model,
        model_provider=model_provider,
        base_url=get_base_url(),
        api_key=get_api_key(),
        # temperature=temperature,
    )
    return chat_model

def new_embbedings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=get_base_url(),
        api_key=get_api_key(),
        model=model,
        check_embedding_ctx_length=False # check_embedding_ctx_length must be set to False for local testing, otherwise it will fail with a 400 error.
    )