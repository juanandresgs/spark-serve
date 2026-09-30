# GLM-5.3-Flash on two Sparks

Our recommended two-Spark GLM recipe serves text and images with an 850K context
window and eight request slots. It distributes one model across both Sparks.

The recipe uses E3 grouped EXL3 experts, BF16 dense layers and adaptive DFlash2
with draft lengths 2, 4 and 7. It retains an 11 GiB FP8 KV budget, a 2,048-token
batch budget, mixed-prefill 512, row cap 128 and the pinned donor transplant.

## Build and deploy

You need two DGX Sparks with working interconnect connectivity, Docker/GPU
support and enough storage for the pinned model and build artifacts. Use your
own hostnames, paths and fabric settings in the site configuration.

Follow [the deployment walkthrough](../../REPRODUCTION.md), selecting
`glm53-flash-adaptive-2spark`. It takes you through site rendering, source staging,
image building, broker installation and API acceptance. `deploy/render_site.py`
produces the site bindings; it does not itself qualify or start the model.

Rebuilt images have new identities. Validate real text, tools, images, context
and all-rank lifecycle behavior on the destination before relying on the service.

## Why adaptive?

In our matched checks, single-request prose increased from 22.22 to 24.98 tokens/s
and eight-request aggregate prose from 72.25 to 88.57. Code was essentially flat
at 55.95 versus 56.04. We retain BF16 dense paths; the public 32.1 tokens/s prose
reference uses FP8 dense paths and is not a matched comparison.

See the [side-by-side tables and public references](../../README.md#why-these-settings)
and [machine-readable evidence](results.json). The evidence is from the original
runtime, not a fresh qualification of the portable package. Recoverable startup
allocation warnings and extended endurance remain open qualification work.
