# Provenance and licenses

The adapter, launcher, harness and documentation source in this directory is
MIT-licensed under [LICENSE](LICENSE). These are independently authored interface
adapters; no TensorFold kernel/model implementation is included in these files.
TensorFold is fetched as a separate, unmodified build dependency.

- [TensorFold](https://github.com/ashhart/TensorFold/tree/34bae79ac97da6c3ab3fe10159cf49633ce8112a),
  exact commit34bae79ac97da6c3ab3fe10159cf49633ce8112a. The exact upstream
  [MIT notice](TensorFold-LICENSE.txt) is retained. Its dependencies retain their
  own licenses. The Docker build retains the source checkout and dependency freeze.
- [Model pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/69e33439ae950f17bcbe95c98f117d80f759ab6d),
  revision69e33439ae950f17bcbe95c98f117d80f759ab6d, branch3.05bpw_h5_ng5.
  Download directly from the publisher and review its model card/license and base
  model terms. We distribute no weights and grant no additional rights to them.
- NVIDIA's digest-pinned PyTorch container remains subject to its publisher's
  terms. No derived binary image is published by this recipe.
- `jsonschema==4.26.0` is installed as a separate dependency. Its dependency graph
  is resolved at build time, not represented as a fully locked environment.

The [improvement survey](IMPROVEMENTS.md) credits primary recipe authors for
mechanisms and independently reported measurements. Those results are not our
measurements; we have not copied their implementation patches. A future kernel
experiment should preserve this separation and establish its own correctness
and matched performance evidence.
