from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceOntology:
    extraction_entities: tuple[str, ...]
    semantic_hints: tuple[str, ...]


def load_source_ontology() -> SourceOntology:
    return SourceOntology(extraction_entities=(), semantic_hints=())
