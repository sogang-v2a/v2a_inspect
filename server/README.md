# v2a-inspect-server

Server-side tooling package for remote visual inference (SAM3, embeddings, label scoring).
This package exists so client-side installs do not need to absorb future heavy runtime dependencies.

Inference is provider-neutral and GPU-backed. Hugging Face is only used as a
weights source during bootstrap; it is not treated as an inference provider.
