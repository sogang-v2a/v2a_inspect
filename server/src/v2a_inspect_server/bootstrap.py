from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field


class WeightsArtifact(BaseModel):
    name: str
    repository: str
    revision: str = "main"
    relative_path: str | None = None
    allow_patterns: list[str] = Field(default_factory=list)


class WeightsManifest(BaseModel):
    artifacts: list[WeightsArtifact] = Field(default_factory=list)


class WeightsBootstrapper:
    def __init__(
        self,
        *,
        cache_dir: Path,
        hf_token: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.hf_token = hf_token

    def load_manifest(self, manifest_path: Path) -> WeightsManifest:
        if not manifest_path.exists():
            return WeightsManifest()
        return WeightsManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    def ensure_manifest(self, manifest: WeightsManifest) -> dict[str, Path]:
        return {
            artifact.name: self.ensure_artifact(artifact)
            for artifact in manifest.artifacts
        }

    def resolve_manifest(self, manifest: WeightsManifest) -> dict[str, Path]:
        return {
            artifact.name: self.resolve_artifact(artifact)
            for artifact in manifest.artifacts
        }

    def ensure_artifact(self, artifact: WeightsArtifact) -> Path:
        destination = self.resolve_artifact(artifact)
        if destination.exists():
            return destination
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN is required to download missing model artifacts."
            )
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=artifact.repository,
            revision=artifact.revision,
            token=self.hf_token,
            local_dir=destination,
            allow_patterns=artifact.allow_patterns or None,
        )
        return destination

    def resolve_artifact(self, artifact: WeightsArtifact) -> Path:
        return self.cache_dir / (artifact.relative_path or artifact.name)
