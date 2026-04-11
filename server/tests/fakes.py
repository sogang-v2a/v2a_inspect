from __future__ import annotations

from dataclasses import dataclass, field

from v2a_inspect.tools.types import (
    CanonicalLabel,
    EntityEmbedding,
    FrameBatch,
    LabelScore,
    Sam3EntityTrack,
    Sam3TrackPoint,
    Sam3TrackSet,
    Sam3VisualFeatures,
)


class FakeSam3Client:
    def extract_entities(
        self,
        frame_batches: list[FrameBatch],
        *,
        prompts_by_scene: dict[int, list[str]] | None = None,
        score_threshold: float = 0.35,
        min_points: int = 2,
        high_confidence_threshold: float = 0.45,
        match_threshold: float = 0.45,
    ) -> Sam3TrackSet:
        del score_threshold, min_points, high_confidence_threshold, match_threshold
        tracks: list[Sam3EntityTrack] = []
        for batch in frame_batches:
            if not batch.frames:
                continue
            label_hint = (
                (prompts_by_scene or {}).get(batch.scene_index, ["object"])[0]
                if prompts_by_scene
                else "object"
            )
            tracks.append(
                Sam3EntityTrack(
                    track_id=f"fake-track-{batch.scene_index}",
                    scene_index=batch.scene_index,
                    start_seconds=batch.frames[0].timestamp_seconds,
                    end_seconds=batch.frames[-1].timestamp_seconds,
                    confidence=0.9,
                    label_hint=label_hint,
                    points=[
                        Sam3TrackPoint(
                            timestamp_seconds=frame.timestamp_seconds,
                            frame_path=frame.image_path,
                            confidence=0.9,
                            bbox_xyxy=[8.0, 8.0, 32.0, 32.0],
                        )
                        for frame in batch.frames
                    ],
                    features=Sam3VisualFeatures(
                        motion_score=0.7,
                        interaction_score=0.2,
                        crowd_score=0.1,
                        camera_dynamics_score=0.1,
                    ),
                )
            )
        return Sam3TrackSet(provider="fake-sam3", strategy="prompt_free", tracks=tracks)


class FakeEmbeddingClient:
    def embed_images(self, image_paths_by_track: dict[str, list[str]]) -> list[EntityEmbedding]:
        return [
            EntityEmbedding(
                track_id=track_id,
                model_name="fake-dinov2",
                vector=[float(index + 1), float(len(image_paths))],
            )
            for index, (track_id, image_paths) in enumerate(sorted(image_paths_by_track.items()))
            if image_paths
        ]


class FakeLabelClient:
    def score_image_labels(self, *, image_paths: list[str], labels: list[str]) -> list[LabelScore]:
        del image_paths
        normalized = [label.strip().lower() for label in labels if label.strip()] or ["object"]
        return [
            LabelScore(label=label, score=max(0.0, 1.0 - index * 0.1))
            for index, label in enumerate(normalized)
        ]

    def score_labels(self, *, group_id: str, image_paths: list[str], labels: list[str]) -> CanonicalLabel:
        scores = self.score_image_labels(image_paths=image_paths, labels=labels)
        return CanonicalLabel(group_id=group_id, label=scores[0].label, scores=scores)


@dataclass(frozen=True)
class FakeToolingRuntime:
    runtime_profile: str = "cpu_dev"
    sam3_client: FakeSam3Client = field(default_factory=FakeSam3Client)
    embedding_client: FakeEmbeddingClient = field(default_factory=FakeEmbeddingClient)
    label_client: FakeLabelClient = field(default_factory=FakeLabelClient)

    @property
    def should_release_clients(self) -> bool:
        return self.runtime_profile == "mig10_safe"

    @property
    def residency_mode(self) -> str:
        return "release_after_stage" if self.should_release_clients else "resident"

    def resident_client_names(self) -> list[str]:
        return ["sam3", "embedding", "label"]


def build_fake_tooling_runtime(*, runtime_profile: str = "cpu_dev") -> FakeToolingRuntime:
    return FakeToolingRuntime(runtime_profile=runtime_profile)
