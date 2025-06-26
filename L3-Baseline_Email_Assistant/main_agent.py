from langgraph.prebuilt import create_react_agent

from lang_graph_project.agent.tools import write_email, schedule_meeting, check_calendar_availability
from lang_graph_project.agent.prompt import create_prompt
from lang_graph_project.utils.open_ai import create_model

class ReactAgent:
    def __init__(self, model:str):
        self.model = model
        # with_structured_output is not implemented for this model and model_provider="ollama"
        # so we use default model_provider="openai" instead
        # llm = create_model(model=model, model_provider="ollama")
        self.chat = create_model(model=model)
        self.tools = [write_email, schedule_meeting, check_calendar_availability]
        self.agent = create_react_agent(
            model=self.chat,
            tools=self.tools,
            prompt=create_prompt,
        )

if __name__ == '__main__':
    # model = "qwen3:1.7b"
    model = "qwen2.5-it:3b"
    react_agent = ReactAgent(model)
    response = react_agent.agent.invoke(
        {"messages": [{
            "role": "user", 
            "content": "what is my availability for tuesday?"
        }]}
    )

    response["messages"][-1].pretty_print()
    print(f"=================================================================================================")
