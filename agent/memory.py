from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from lang_graph_project.utils import open_ai

def new_manage_memory_tool():
    return create_manage_memory_tool(
        namespace=(
            "email_assistant", 
            "{langgraph_user_id}",
            "collection"
        )
    )

def new_search_memory_tool():
    return create_search_memory_tool(
        namespace=(
            "email_assistant",
            "{langgraph_user_id}",
            "collection"
        )
    )

def new_in_store_memory(model:str) -> InMemoryStore:
    # since we don't use ollama(openai compatible style), we can't use InMemoryStore with index directly:
    # since some of the parameters such as provider, api_key, base_url are initialized by default.
    # but according to the InMemoryStore doc, we can use it an embedding initialized by ourselves.
    # store = InMemoryStore(
    #     index={"embed": "openai:text-embedding-3-small"},
    # )
    return InMemoryStore(
        index={
            "embed": open_ai.new_embbedings(model=model),
        }
    )

if __name__ == "__main__":
    model="deepseek-r1:1.5b"
    store = new_in_store_memory(model=model)

    manage_memory_tool = new_manage_memory_tool()
    search_memory_tool = new_search_memory_tool()

    print("manage_memory_tool:\n")
    print(f"name: \n{manage_memory_tool.name}")
    print(f"description: \n{manage_memory_tool.description}")
    print(f"arguments: \n{manage_memory_tool.args}\n")
    print(f"=================================================================================================")

    print("search_memory_tool:\n")
    print(f"name: \n{search_memory_tool.name}")
    print(f"description: \n{search_memory_tool.description}")
    print(f"arguments: \n{search_memory_tool.args}\n")
    print(f"=================================================================================================")
