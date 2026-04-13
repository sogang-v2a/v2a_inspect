from __future__ import annotations

from v2a_inspect.pipeline.prompt_templates import ResolvedPrompt


def append_prompt_evidence(
    prompt: ResolvedPrompt,
    *,
    title: str,
    evidence: str,
) -> ResolvedPrompt:
    block = f"\n\n[{title}]\n{evidence.strip()}"
    return ResolvedPrompt(
        name=prompt.name,
        system_text=prompt.system_text,
        user_text=f"{prompt.user_text}{block}",
        source=prompt.source,
        langfuse_prompt=prompt.langfuse_prompt,
    )
