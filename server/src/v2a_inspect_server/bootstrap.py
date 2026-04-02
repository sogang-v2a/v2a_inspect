from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .remote import download_file


class WeightsArtifact(BaseModel):
    name: str
    repository: str
    filename: str
    revision: str = "main"
    relative_path: str | None = None


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

    def ensure_artifact(self, artifact: WeightsArtifact) -> Path:
        destination = self.cache_dir / (artifact.relative_path or artifact.filename)
        if destination.exists():
            return destination
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN is required to download missing model artifacts."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_file(
            self.build_download_url(artifact),
            destination,
            api_key=self.hf_token,
        )
        return destination

    @staticmethod
    def build_download_url(artifact: WeightsArtifact) -> str:
        return (
            f"https://huggingface.co/{artifact.repository}/resolve/"
            f"{artifact.revision}/{artifact.filename}"
        )
