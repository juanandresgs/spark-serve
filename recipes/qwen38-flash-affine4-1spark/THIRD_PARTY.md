# Provenance

Our interface adapters, reader, launch script and harness are independently authored
and licensed under LICENSE. No Mia implementation, container or launcher is used.
The mechanism descriptions and separately attributed performance claims informed
experiments. No upstream inference/kernel implementation is copied into this kit.

TensorFold is fetched as an unmodified dependency at
71377a5373ed7b394f1b480ba2a6a3986b03af1c (0.3.6.2). Its MIT notice is included.
We deliberately retain this version: later 0.3.6.3 credits Mia-contributed typed
arguments; this kit retains our own typed adapter. Upgrading needs requalification.

Download the Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP pack directly, revision
dadefa8066e3be900a0d148d0f5a2f4eb1cf6534. Model/base-model and NVIDIA container
terms remain their publishers' terms. No weights or binary image are distributed.
jsonschema 4.26.0 and pybind11 3.0.4 are separately installed dependencies.
The base, runtime and model revisions are pinned; transitive pip dependencies
are resolved at build time and captured in the image's dependency freeze.
A rebuilt image need not have the same digest as the measured production image.
