from importlib import resources

assert isinstance(__package__, str), "Expected __package__ to be a string"
BASE_PATH = f"{__package__}.prompts.user"


def _get_prompt(prompt_name: str) -> str:
    """Load a user prompt template from package resources."""

    prompt_name = prompt_name.lower().replace(" ", "_").replace("-", "_").strip()
    prompt_dir = resources.files(BASE_PATH)
    return prompt_dir.joinpath(f"{prompt_name}.txt").read_text(encoding="utf-8")
