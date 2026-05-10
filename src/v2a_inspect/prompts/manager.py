from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v2a_inspect.config import settings
from v2a_inspect.prompts.errors import PromptNotFoundError, PromptRenderError


@dataclass(frozen=True)
class PromptPair:
    name: str
    system: str
    user: str


class PromptManager:
    def __init__(
        self,
        prompts_dir: Path | str | None = None,
        file_suffix: str | None = None,
    ) -> None:
        self.prompts_dir = Path(prompts_dir or settings.prompts_dir)
        self.file_suffix = file_suffix or settings.prompt_file_suffix

    def list_prompts(self) -> list[str]:
        return sorted(self._names_in("system") & self._names_in("user"))

    def get_prompt(self, name: str) -> PromptPair:
        normalized = self._normalize_name(name)
        system_path = self._path_for_name("system", normalized)
        user_path = self._path_for_name("user", normalized)
        if not system_path.is_file() or not user_path.is_file():
            raise PromptNotFoundError(f"Prompt pair not found: {normalized}")

        return PromptPair(
            name=normalized,
            system=system_path.read_text(encoding="utf-8"),
            user=user_path.read_text(encoding="utf-8"),
        )

    def render_prompt(self, name: str, **values: Any) -> PromptPair:
        prompt = self.get_prompt(name)
        try:
            return PromptPair(
                name=prompt.name,
                system=prompt.system.format(**values),
                user=prompt.user.format(**values),
            )
        except KeyError as exc:
            missing_key = exc.args[0]
            raise PromptRenderError(
                f"Missing value for prompt variable '{missing_key}' in prompt: {name}"
            ) from exc

    def _path_for_name(self, role: str, name: str) -> Path:
        return self.prompts_dir / role / f"{name}{self.file_suffix}"

    def _names_in(self, role: str) -> set[str]:
        root = self.prompts_dir / role
        if not root.exists():
            return set()

        names = set()
        for path in root.rglob(f"*{self.file_suffix}"):
            if path.is_file():
                names.add(self._name_from_path(root, path))
        return names

    def _name_from_path(self, root: Path, path: Path) -> str:
        relative = path.relative_to(root)
        return relative.with_suffix("").as_posix()

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip().removesuffix(".txt")
        if not normalized or "/" in normalized or ".." in normalized:
            raise PromptNotFoundError("Prompt name cannot be empty")
        return normalized
