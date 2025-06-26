import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image, display

from lang_graph_project.schemas.state import State
from lang_graph_project.constants.prompt_templates import triage_system_prompt_template, triage_user_prompt_template
from lang_graph_project.constants.variables import profile, prompt_instructions
from lang_graph_project.utils.formator import format_few_shot_examples_v1

import base_triag
import main_agent

class EmailAgent():
    def __init__(self, 
                 triage_agent: base_triag.TriageAgent, 
                 main_agent: main_agent.ReactAgent):
        self.triage_agent = triage_agent
        self.main_agent = main_agent
        email_agent = StateGraph(State)
        email_agent = email_agent.add_node("triage_router", self.triage_router)
        email_agent = email_agent.add_node("response_agent", self.main_agent.agent)
        email_agent = email_agent.add_edge(START, "triage_router")
        # langgraph.StateGraph.compile has some changes on compile parameters
        # so the following code commented below cannot be run correctly.
        # email_agent = email_agent.compile(store)
        email_agent = email_agent.compile(store=self.main_agent.store)
        self.email_agent = email_agent

    def triage_router(self, state: State, config) -> Command[
        Literal["response_agent", "__end__"]
    ]:
        author = state['email_input']['author']
        to = state['email_input']['to']
        subject = state['email_input']['subject']
        email_thread = state['email_input']['email_thread']
        user_prompt = triage_user_prompt_template.format(
            author=author, 
            to=to, 
            subject=subject, 
            email_thread=email_thread
            )
        
        namespace = (
            "email_assistant",
            config['configurable']['langgraph_user_id'],
            "examples"
        )
        examples = self.main_agent.store.search(
            namespace, 
            query=str({"email": state['email_input']})
        ) 
        examples=format_few_shot_examples_v1(examples)

        full_name = profile['full_name']
        name = profile['name']
        triag_rules = prompt_instructions['triage_rules']
        user_profile_background = profile['user_profile_background']
        triage_ignore = triag_rules['ignore']
        triage_notify = triag_rules['notify']
        traige_email = triag_rules['respond']
        system_prompt = triage_system_prompt_template.format(
            full_name=full_name,
            name=name,
            user_profile_background=user_profile_background,
            triage_no=triage_ignore,
            triage_notify=triage_notify,
            triage_email=traige_email,
            examples=None
            )
        
        result = self.triage_agent.llm_router.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        if result.classification == "respond":
            print("📧 Classification: RESPOND - This email requires a response")
            goto = "response_agent"
            update = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Respond to the email {state['email_input']}",
                    }
                ]
            }
        elif result.classification == "ignore":
            print("🚫 Classification: IGNORE - This email can be safely ignored")
            update = None
            goto = END
        elif result.classification == "notify":
            # If real life, this would do something else
            print("🔔 Classification: NOTIFY - This email contains important information")
            update = None
            goto = END
        else:
            raise ValueError(f"Invalid classification: {result.classification}")
        return Command(goto=goto, update=update)

if __name__ == "__main__":
    model = "qwen2.5-it:3b"
    agent = EmailAgent(
        base_triag.TriageAgent(model), 
        main_agent.ReactAgent(model=model), 
    )    
    # Show the agent
    display(Image(agent.email_agent.get_graph(xray=True).draw_mermaid_png()))
    
    # Useless currently
    # config = {"configurable": {"langgraph_user_id": "lance"}}

    # first add some examples to memory to parse
    # example 1
    email = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John,

    I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

    Specifically, I'm looking at:
    - /auth/refresh
    - /auth/validate

    Thanks!
    Alice""",
    }

    data = {
        "email": email,
        # This is to start changing the behavior of the agent
        "label": "respond"
    }

    agent.main_agent.store.put(
        ("email_assistant", "lance", "examples"), 
        str(uuid.uuid4()), 
        data
    )

    # example 2
    data = {
        "email": {
            "author": "Sarah Chen <sarah.chen@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Update: Backend API Changes Deployed to Staging",
            "email_thread": """Hi John,
        
        Just wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:
        
        - Implemented JWT refresh token rotation
        - Added rate limiting for login attempts
        - Updated API documentation with new endpoints
        
        All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*
        
        No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.
        
        Best regards,
        Sarah
        """,
        },
        "label": "ignore"
    }

    agent.main_agent.store.put(
        ("email_assistant", "lance", "examples"),
        str(uuid.uuid4()),
        data
    )

    email_data = {
            "author": "Sarah Chen <sarah.chen@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Update: Backend API Changes Deployed to Staging",
            "email_thread": """Hi John,
        
        Wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:
        
        - Implemented JWT refresh token rotation
        - Added rate limiting for login attempts
        - Updated API documentation with new endpoints
        
        All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*
        
        No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.
        
        Best regards,
        Sarah
        """,
        }
    results = agent.main_agent.store.search(
            ("email_assistant", "lance", "examples"),
            query=str({"email": email_data}),
            limit=1
        )
    print(f"examples: {format_few_shot_examples_v1(results)}")

    email_input = {
        "author": "Tom Jones <tome.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John - want to buy documentation?""",
    }
    response = agent.email_agent.invoke(
        {"email_input": email_input}, 
        config={"configurable": {"langgraph_user_id": "harrison"}}
    )
    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")

    # Update store to ignore emails like this
    data = {
        "email": {
            "author": "Tom Jones <tome.jones@bar.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Quick question about API documentation",
            "email_thread": """Hi John - want to buy documentation?""",
        },
        "label": "ignore"
    }
    agent.main_agent.store.put(
        ("email_assistant", "harrison", "examples"),
        str(uuid.uuid4()),
        data
    )

    # Try it again, it should ignore this time
    email_input = {
        "author": "Tom Jones <tome.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John - want to buy documentation?""",
    }

    response = agent.email_agent.invoke(
        {"email_input": email_input}, 
        config={"configurable": {"langgraph_user_id": "harrison"}}
    )
    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")

    # Slightly modify text, will continue to ignore
    email_input = {
        "author": "Jim Jones <jim.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John - want to buy documentation?????""",
    }

    response = agent.email_agent.invoke(
        {"email_input": email_input}, 
        config={"configurable": {"langgraph_user_id": "harrison"}}
    )
    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")

    # Try with a different user id
    response = agent.email_agent.invoke(
        {"email_input": email_input}, 
        config={"configurable": {"langgraph_user_id": "andrew"}}
    )
    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")
