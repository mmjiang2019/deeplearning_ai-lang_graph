import openai

from typing import List

# from typing import List, Tuple
# from dotenv import load_dotenv, find_dotenv

from pair_programming.config import open_ai as open_ai_config

def NewOpenAIClient():
    # _ = load_dotenv(find_dotenv()) # read local .env file.
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(
        base_url=open_ai_config.BASE_URL,
        api_key=open_ai_config.API_KEY,  # this is also the default, it can be omitted
        )
    return client

def get_base_url() -> str:
    return open_ai_config.BASE_URL

def get_api_key() -> str:
    return open_ai_config.API_KEY

def list_models(client: openai.OpenAI):
    return client.models.list()

def select_model(client: openai.OpenAI, prefixes: List[str]):
    models = []
    for m in list_models(client=client):
        for prefix in prefixes:
            if m.id.startswith(prefix):
                models.append(m)
    return models

def get_completion(client: openai.OpenAI, prompt: str, model: str):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content