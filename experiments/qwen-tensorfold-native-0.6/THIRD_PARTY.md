# Third-party notices

- TensorFold is fetched from the official `ashhart/TensorFold` repository at the immutable revision in `pins.json`. Its upstream MIT license is retained in `TensorFold-LICENSE.txt`.
- The base is the NVIDIA PyTorch container pinned by digest in `pins.json`. NVIDIA container and component terms apply; consult the image's own notices.
- Model weights are published separately by `turboderp` at the immutable repository revision in `pins.json`. Consult that repository's model card and license before downloading or redistributing weights.
- The optional grammar image installs XGrammar, Transformers, Hugging Face Hub, hf-xet, Typer, Shellingham, and annotated-doc at the exact versions in `grammar-requirements.txt`. Their wheel license metadata is retained in the image; review each distribution's notices before redistribution. XGrammar is published under Apache-2.0 by [MLC AI](https://github.com/mlc-ai/xgrammar).
- This scaffold contains no copied private deployment implementation.
