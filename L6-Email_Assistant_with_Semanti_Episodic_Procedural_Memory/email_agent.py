import json
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image, display
from langgraph.config import get_store
from langmem import create_multi_prompt_optimizer

from lang_graph_project.schemas.state import State
from lang_graph_project.constants.prompt_templates import triage_system_prompt_template, triage_user_prompt_template
from lang_graph_project.constants.variables import profile, prompt_instructions
from lang_graph_project.utils.formator import format_few_shot_examples_v1

import triage_agent
import main_agent

class EmailAgent():
    def __init__(self, 
                 triage_agent: triage_agent.TriageAgent, 
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
        store = get_store()
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

        langgraph_user_id = config['configurable']['langgraph_user_id']
        namespace = (langgraph_user_id, )

        triag_rules = prompt_instructions['triage_rules']
        triage_ignore = triag_rules['ignore']
        triage_notify = triag_rules['notify']
        traige_email = triag_rules['respond']
        
        result = store.get(namespace, "triage_ignore")
        if result is None:
            store.put(
                namespace, 
                "triage_ignore", 
                {"prompt": triage_ignore}
            )
            ignore_prompt = triage_ignore
        else:
            ignore_prompt = result.value['prompt']
        result = store.get(namespace, "triage_notify")
        if result is None:
            store.put(
                namespace, 
                "triage_notify", 
                {"prompt": triage_notify}
            )
            notify_prompt = triage_notify
        else:
            notify_prompt = result.value['prompt']

        result = store.get(namespace, "triage_respond")
        if result is None:
            store.put(
                namespace, 
                "triage_respond", 
                {"prompt": traige_email}
            )
            respond_prompt = traige_email
        else:
            respond_prompt = result.value['prompt']
        
        full_name = profile['full_name']
        name = profile['name']
        user_profile_background = profile['user_profile_background']

        system_prompt = triage_system_prompt_template.format(
            full_name=full_name,
            name=name,
            user_profile_background=user_profile_background,
            triage_no=ignore_prompt,
            triage_notify=notify_prompt,
            triage_email=respond_prompt,
            examples=examples
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
        triage_agent.TriageAgent(model), 
        main_agent.ReactAgent(model=model), 
    )    
    # Show the agent
    display(Image(agent.email_agent.get_graph(xray=True).draw_mermaid_png()))
    
    # Useless currently
    config = {"configurable": {"langgraph_user_id": "lance"}}

    email_input = {
        "author": "Alice Jones <alice.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John,

    Urgent issue - your service is down. Is there a reason why""",
    }

    response = agent.email_agent.invoke(
        {"email_input": email_input},
        config=config
    )
    for m in response["messages"]:
        m.pretty_print()
    print(f"=================================================================================================")

    print(f"prompts for lance in memory: ")
    store = agent.main_agent.store
    namespace = ("lance",)
    agent_instructions = store.get(namespace, "agent_instructions").value['prompt']
    triage_respond = store.get(namespace, "triage_respond").value['prompt']
    triage_ignore = store.get(namespace, "triage_ignore").value['prompt']
    triage_notify = store.get(namespace, "triage_notify").value['prompt']
    mem_prompts = {
        "agent_instructions": agent_instructions,
        "triage_respond": triage_respond,
        "triage_ignore": triage_ignore,
        "triage_notify": triage_notify,
    }
    print(f"mem_prompts: \n{mem_prompts}")
    print(f"=================================================================================================")

    # User a LLM to update instructions
    # TODO: update and wrap the logic here

    # Update main_agent prompt
    conversations = [
        (
            response['messages'],
            "Always sign your emails `John Doe`"
        )
    ]

    prompts = [
        {
            "name": "main_agent",
            "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
            
        },
        {
            "name": "triage-ignore", 
            "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

        },
        {
            "name": "triage-notify", 
            "prompt": store.get(("lance",), "triage_notify").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

        },
        {
            "name": "triage-respond", 
            "prompt": store.get(("lance",), "triage_respond").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

        },
    ]

    optimizer = create_multi_prompt_optimizer(
        agent.main_agent.chat,
        kind="prompt_memory",
    )

    updated = optimizer.invoke(
        {"trajectories": conversations, "prompts": prompts}
    )
    print(updated)
    #json dumps is a bit easier to read
    print(json.dumps(updated, indent=4))

    for i, updated_prompt in enumerate(updated):
        old_prompt = prompts[i]
        if updated_prompt['prompt'] != old_prompt['prompt']:
            name = old_prompt['name']
            print(f"updated {name}")
            if name == "main_agent":
                store.put(
                    ("lance",),
                    "agent_instructions",
                    {"prompt":updated_prompt['prompt']}
                )
            else:
                #raise ValueError
                print(f"Encountered {name}, implement the remaining stores!")

    store.get(("lance",), "agent_instructions").value['prompt']

    response = agent.email_agent.invoke(
        {"email_input": email_input}, 
        config=config
    )

    for m in response["messages"]:
        m.pretty_print()

    email_input = {
        "author": "Alice Jones <alice.jones@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John,

    Urgent issue - your service is down. Is there a reason why""",
    }

    response = agent.email_agent.invoke(
        {"email_input": email_input},
        config=config
    )

    # Update triage-ignore prompt
    conversations = [
        (
            response['messages'],
            "Ignore any emails from Alice Jones"
        )
    ]
    prompts = [
        {
            "name": "main_agent",
            "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
            
        },
        {
            "name": "triage-ignore", 
            "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

        },
        {
            "name": "triage-notify", 
            "prompt": store.get(("lance",), "triage_notify").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

        },
        {
            "name": "triage-respond", 
            "prompt": store.get(("lance",), "triage_respond").value['prompt'],
            "update_instructions": "keep the instructions short and to the point",
            "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

        },
    ]

    updated = optimizer.invoke(
        {"trajectories": conversations, "prompts": prompts}
    )

    for i, updated_prompt in enumerate(updated):
        old_prompt = prompts[i]
        if updated_prompt['prompt'] != old_prompt['prompt']:
            name = old_prompt['name']
            print(f"updated {name}")
            if name == "main_agent":
                store.put(
                    ("lance",),
                    "agent_instructions",
                    {"prompt":updated_prompt['prompt']}
                )
            if name == "triage-ignore":
                store.put(
                    ("lance",),
                    "triage_ignore",
                    {"prompt":updated_prompt['prompt']}
                )
            else:
                #raise ValueError
                print(f"Encountered {name}, implement the remaining stores!")

    response = agent.email_agent.invoke(
        {"email_input": email_input},
        config=config
    )
    store.get(("lance",), "triage_ignore").value['prompt']