
from lang_graph_project.schemas import router
from lang_graph_project.utils.open_ai import create_model

from lang_graph_project.constants.prompt_templates import triage_system_prompt_template, triage_user_prompt_template
from lang_graph_project.constants.variables import profile, prompt_instructions

class TriageAgent:
    def __init__(self, model:str):
        self.model = model
        # with_structured_output is not implemented for this model and model_provider="ollama"
        # so we use default model_provider="openai" instead
        # llm = create_model(model=model, model_provider="ollama")
        self.chat = create_model(model=model)
        self.llm_router = self.chat.with_structured_output(router.Router)

if __name__ == "__main__":

    # Example incoming email
    email = {
        "from": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "body": """
    Hi John,

    I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

    Specifically, I'm looking at:
    - /auth/refresh
    - /auth/validate

    Thanks!
    Alice""",
    }

    model = "qwen2.5-it:3b"

    tiage_agent = TriageAgent(model)

    system_prompt = triage_system_prompt_template.format(
        full_name=profile["full_name"],
        name=profile["name"],
        examples=None,
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
    )
    print(f"system_prompt:\n{system_prompt}\n")
    print(f"=================================================================================================")

    user_prompt = triage_user_prompt_template.format(
        author=email["from"],
        to=email["to"],
        subject=email["subject"],
        email_thread=email["body"],
    )

    print(f"user_prompt:\n{user_prompt}\n")
    print(f"=================================================================================================")

    result = tiage_agent.llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    print(f"result:\n{result}")
    print(f"=================================================================================================")
