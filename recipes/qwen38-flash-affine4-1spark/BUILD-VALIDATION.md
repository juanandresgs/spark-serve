# Build and deployment validation

On September30, the measured Affine4 image34a5ffedbb9c was selected for production.
The named production endpoint passed JSON/API11/11 and typed-tool/API36/36 checks,
plus native Pi and Hermes typed-tool/result round trips. The four-slot text
contract remains unchanged, with no global forced reasoning cap.

The standalone Dockerfile was independently built from the public pinned base,
TensorFold revision and included adapter sources. Build image:
`sha256:00798fd9f2c752ab20d62bd31a00817950cf0835276d1e8af3c82cbb4859c286`.
It compiled its CUDA kernels and loaded the model on a spare DGX Spark. It passed
HTTP11/11, native typed-tool/API36/36 and six sampled reasoning replays with exact
token parity to the qualified production artifact. The image also passed10 adapter
unit tests and the real-file positional-reader equality/error checks. It was
stopped afterward. See build-validation.json for machine-readable scope.

The existing pinned, checksum-verified model files were reused; a new checkpoint
download on a clean machine was not repeated. Stage verification logic is included.
Full context/throughput/296-question quality measurements belong to the qualified
image, not this source rebuild. The eight exported adapter files match that image's
bytes. Base/model/runtime revisions are pinned, but transitive pip resolution and
build metadata can change the rebuilt image digest. No binary image is included.

The private deployment/export repository passed375 tests, including privacy and
manifest integrity checks. The initial standalone build uncovered a Docker ignore
allowlist omission; it was fixed before the successful build above. No private
source path or base image is needed to follow README.md.
