from lang_graph_project.constants.prompt_templates import agent_system_prompt_template
from lang_graph_project.constants.variables import prompt_instructions, profile

def create_prompt(state):
    return [
        {
            "role": "system",
            "content": agent_system_prompt_template.format(
                instructions=prompt_instructions['agent_instructions'],
                **profile,
            ),
        }
    ] + state['messages']