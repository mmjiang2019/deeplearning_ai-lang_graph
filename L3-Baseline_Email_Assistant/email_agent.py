from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image, display

from lang_graph_project.schemas.state import State
from lang_graph_project.constants.prompt_templates import triage_system_prompt_template, triage_user_prompt_template
from lang_graph_project.constants.variables import profile, prompt_instructions

import base_triag
import main_agent

class EmailAgent():
    def __init__(self, triage_agent: base_triag.TriageAgent, main_agent: main_agent.ReactAgent):
        self.triage_agent = triage_agent
        self.main_agent = main_agent
        email_agent = StateGraph(State)
        email_agent = email_agent.add_node(self.triage_router)
        email_agent = email_agent.add_node("response_agent", self.main_agent.agent)
        email_agent = email_agent.add_edge(START, "triage_router")
        email_agent = email_agent.compile()
        self.email_agent = email_agent

    def triage_router(self, state: State) -> Command[
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
    agent = EmailAgent(base_triag.TriageAgent(model), main_agent.ReactAgent(model=model))
    
    # Show the agent
    display(Image(agent.email_agent.get_graph(xray=True).draw_mermaid_png()))

    email_input = {
        "author": "Marketing Team <marketing@amazingdeals.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
        "email_thread": """Dear Valued Developer,

    Don't miss out on this INCREDIBLE opportunity! 

    🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite! 

    ✨ FEATURES:
    - Revolutionary AI-powered code completion
    - Cloud-based development environment
    - 24/7 customer support
    - And much more!

    💰 Regular Price: $999/month
    🎉 YOUR SPECIAL PRICE: Just $199/month!

    🕒 Hurry! This offer expires in:
    24 HOURS ONLY!

    Click here to claim your discount: https://amazingdeals.com/special-offer

    Best regards,
    Marketing Team
    ---
    To unsubscribe, click here
    """,
    }
    response = agent.email_agent.invoke({"email_input": email_input})
    print(f"=================================================================================================")

    email_input = {
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
    response = agent.email_agent.invoke({"email_input": email_input})
    print(f"=================================================================================================")

    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")
