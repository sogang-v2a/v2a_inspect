from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from v2a_inspect.config import settings
from v2a_inspect.prompts.errors import PromptNotFoundError


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

    def get_prompt(self, name: str) -> ChatPromptTemplate:
        normalized = self._normalize_name(name)
        system_path = self._path_for_name("system", normalized)
        user_path = self._path_for_name("user", normalized)
        if not system_path.is_file() or not user_path.is_file():
            raise PromptNotFoundError(f"Prompt template not found: {normalized}")

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_path.read_text(encoding="utf-8")),
                ("human", user_path.read_text(encoding="utf-8")),
            ]
        )

    def get_multimodal_prompt(
        self,
        name: str,
        content_blocks: list[dict[str, Any]],
    ) -> ChatPromptTemplate:
        normalized = self._normalize_name(name)
        system_path = self._path_for_name("system", normalized)
        user_path = self._path_for_name("user", normalized)
        if not system_path.is_file() or not user_path.is_file():
            raise PromptNotFoundError(f"Prompt template not found: {normalized}")

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_path.read_text(encoding="utf-8")}
        ]
        user_content.extend(content_blocks)
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_path.read_text(encoding="utf-8")),
                ("human", user_content),
            ]
        )

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
