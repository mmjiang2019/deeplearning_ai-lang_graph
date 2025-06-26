from langgraph.prebuilt import create_react_agent

from lang_graph_project.agent.tools import write_email, schedule_meeting, check_calendar_availability
from lang_graph_project.agent.prompt import create_prompt_with_memory
from lang_graph_project.agent.memory import new_manage_memory_tool, new_search_memory_tool, new_in_store_memory
from lang_graph_project.utils.open_ai import create_model

class ReactAgent:
    def __init__(self, model:str):
        self.model = model
        # with_structured_output is not implemented for this model and model_provider="ollama"
        # so we use default model_provider="openai" instead
        # llm = create_model(model=model, model_provider="ollama")
        self.chat = create_model(model=model)
        self.store = new_in_store_memory(model=model)
        self.manage_memory_tool = new_manage_memory_tool()
        self.search_memory_tool = new_search_memory_tool()
        self.tools = [
            write_email, 
            schedule_meeting, 
            check_calendar_availability,
            self.manage_memory_tool,
            self.search_memory_tool,
            ]
        self.agent = create_react_agent(
            model=self.chat,
            tools=self.tools,
            prompt=create_prompt_with_memory,
            # Use this to ensure the store is passed to the agent 
            store = self.store,
        )

# Codes below are moved to class ReactAgent
# llm = create_model(model=model)

# tools=[write_email, schedule_meeting, check_calendar_availability]
# agent = create_react_agent(
#     model=llm,
#     tools=tools,
#     prompt=create_prompt,
# )

if __name__ == '__main__':
    # model = "qwen3:1.7b"
    model = "qwen2.5-it:3b"
    react_agent = ReactAgent(model)
    config = {"configurable": {"langgraph_user_id": "lance"}}
    response = react_agent.agent.invoke(
        {"messages": [{
            "role": "user", 
            "content": "Jim is John's friend."
        }]},
        config=config,
    )

    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")

    response = react_agent.agent.invoke(
        {"messages": [{"role": "user", "content": "who is jim?"}]},
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")

    print(f"namespaces in store: \n{react_agent.store.list_namespaces()}")
    search_result = react_agent.store.search(('email_assistant', 'lance', 'collection'))
    print(f"search result:\n {search_result}")

    search_result = react_agent.store.search(('email_assistant', 'lance', 'collection'), query="jim")
    print(f"search result with query 'jim':\n {search_result}")
