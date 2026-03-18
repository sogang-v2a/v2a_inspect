from importlib import resources
from langchain_core.prompts import ChatPromptTemplate

assert isinstance(__package__, str), "Expected __package__ to be a string"
BASE_PATH = f"{__package__}.prompts"


def _get_prompt(prompt_name: str) -> ChatPromptTemplate:
    """Load prompt template from package resources."""

    prompt_name = prompt_name.lower().replace(" ", "_").replace("-", "_").strip()

    human_dir = resources.files(f"{BASE_PATH}.user")
    system_dir = resources.files(f"{BASE_PATH}.system")

    human_prompt_str = human_dir.joinpath(f"{prompt_name}.txt").read_text(
        encoding="utf-8"
    )
    system_prompt_str = system_dir.joinpath(f"{prompt_name}.txt").read_text(
        encoding="utf-8"
    )

    return ChatPromptTemplate(
        [
            ("system", system_prompt_str),
            ("human", human_prompt_str),
        ]
    )
