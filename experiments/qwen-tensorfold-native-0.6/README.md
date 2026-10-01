# TensorFold 0.6.0 native source build (experimental)

This is a public, portable source-build scaffold for TensorFold 0.6.0 and the pinned Qwen EXL3 model. It is retained for evaluation only. It is not the production recommendation, and this public source kit has not yet been rebuilt on a DGX Spark or qualified for GPU behavior, model quality, throughput, or serving reliability.

The base image, runtime commit, and model revision are pinned in `pins.json`. The base image's 216-package inventory and the tested primary runtime's 217-package inventory differ only by TensorFold 0.6.0. `runtime-constraints.txt` pins the base inventory; it is not an install requirements file. The Dockerfile installs the pinned TensorFold source without dependency resolution, checks that every active TensorFold dependency is present and satisfies its declared version, then compares the complete result with `expected-primary-inventory.json`. A build fails for missing packages, extra packages, or version drift. Capture method and package deltas are recorded in [`inventory-provenance.json`](inventory-provenance.json).

## Build and inspect

On a Linux ARM64 host with Docker and access to the pinned NVIDIA base image:

```sh
cd experiments/qwen-tensorfold-native-0.6
python3 build.py --tag local/qwen-tensorfold-native:0.6.0 \
  --receipt ./artifacts/build-receipt.json
```

The build pulls the exact base digest, checks out the exact TensorFold commit, validates the runtime version and complete package inventory, then records the local image ID, repository digests when available, rootfs diff IDs, source hashes, and pinned model/runtime identity. Image IDs are daemon-store-specific: compare repository and rootfs provenance across Docker stores rather than requiring `.Id` equality. The local `.Id` is not treated as a config digest.

This command builds the primary runtime only. It does not download model weights, create a serving configuration, or deploy anything. Use a separately reviewed model staging and launch procedure. Do not treat a successful build or matching package inventory as GPU or end-to-end qualification.

## Optional grammar image

Grammar support is a separate image and remains unqualified. Its pinned package delta is recorded in `grammar-requirements.txt`, with the complete tested 222-package inventory in `expected-grammar-inventory.json`. Build it only from the exact local image ID produced by the primary build:

```sh
PARENT_IMAGE_ID="$(python3 -c 'import json; print(json.load(open("./artifacts/build-receipt.json"))["local_image_id"])')"
python3 build_grammar.py --parent-image-id "$PARENT_IMAGE_ID" \
  --tag local/qwen-tensorfold-native:0.6.0-grammar \
  --receipt ./artifacts/grammar-build-receipt.json
```

The builder checks the requested local ID, binds a unique temporary Docker tag to that ID, verifies the binding before building, and records the parent and child rootfs identities. The grammar image's package inventory must exactly match its tested inventory. This build does not establish grammar-constrained generation correctness or tool compatibility.

## Qualification boundary

The scaffold still needs an actual build from this exported public kit on the target Spark, followed by model load and API/tool checks. Quality, speed, full-context behavior, restart/reboot recovery, endurance, and optional grammar support are unmeasured here. Grammar support is deliberately not included in this base image. No performance or recommendation claim is made by this experimental recipe.

See the [recipe qualification record](../../recipes/qwen-tensorfold-native-exl3/qualification.json) and [third-party notices](THIRD_PARTY.md).
