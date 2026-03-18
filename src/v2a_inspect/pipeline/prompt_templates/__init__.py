from .utils import _get_prompt

GROUPING_PROMPT_TEMPLATE = _get_prompt("grouping")
MODEL_SELECT_PROMPT_TEMPLATE = _get_prompt("model_select")
SCENE_ANALYSIS_DEFAULT_PROMPT_TEMPLATE = _get_prompt("scene_analysis_default")
SCENE_ANALYSIS_EXTENDED_PROMPT_TEMPLATE = _get_prompt("scene_analysis_extended")
VLM_VERIFY_PROMPT_TEMPLATE = _get_prompt("vlm_verify")

__all__ = [
    "GROUPING_PROMPT_TEMPLATE",
    "MODEL_SELECT_PROMPT_TEMPLATE",
    "SCENE_ANALYSIS_DEFAULT_PROMPT_TEMPLATE",
    "SCENE_ANALYSIS_EXTENDED_PROMPT_TEMPLATE",
    "VLM_VERIFY_PROMPT_TEMPLATE",
]
