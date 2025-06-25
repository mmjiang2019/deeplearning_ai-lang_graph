from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]