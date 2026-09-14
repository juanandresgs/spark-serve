# Deploying the pinned recipes

The following path is for ARM64 Linux DGX Sparks with Docker and NVIDIA Container
Toolkit already working. Use Python 3.11+, Bash, Git, patch, rsync, SSH, the Hugging
Face `hf` CLI, and enough local disk for the selected model on every rank.
The service account needs Docker access and passwordless SSH to the site aliases.
Host keys must already be trusted. The release does not create accounts or keys.

Keep a source checkout and build outputs on durable local storage. Never put
weights, credentials or a filled-in site file in the public repository.

Before staging, verify the source package and offline site binding:

```sh
PYTHONPATH=src python3 deploy/check.py
```

This checks every manifested file, all recipe assets and two sample site bindings
for each portable adapter. It does not access GPUs or download model weights.

## 1. Fetch and build

Fetch each required source using its immutable revision:

```sh
python3 reproduce.py fetch glm-e2 --output source-checkouts/glm-e2
python3 reproduce.py fetch glm-e3 --output source-checkouts/glm-e3
python3 reproduce.py fetch qwen-single --output source-checkouts/qwen-single
python3 reproduce.py fetch llama-swap --output source-checkouts/llama-swap
```

For adaptive GLM, prepare the image chain, then compile on an idle Spark:

```sh
python3 deploy/build_images.py prepare glm-adaptive --sources source-checkouts --output build/adaptive
python3 deploy/build_images.py execute build/adaptive/build-plan.json
```

The chain builds E2 from the pinned public base, compiles only the reviewed E3
grouped kernels, then applies the exact adaptive scheduler/graph patch. It
does not adopt the later upstream FP8-dense or template changes. The final
image reference and actual local image ID are written into `build-receipt.json`.
Build once and distribute with `docker save` / `docker load`, or build on each
rank. Record the loaded ID on every node; image IDs can differ between Docker
storage representations. Never substitute an old measured image ID for a new one.

Qwen TP1 needs no local compilation. Pull its public image from the recipe
profile, using the complete `@sha256` reference, on each Qwen node. Record
`docker image inspect --format '{{.Id}}' IMAGE` for the site file.

The router needs Go 1.26.5 and Node/npm. Build on Linux or macOS; the output is
Linux ARM64. This includes the native `ready` strategy needed for two replicas:

```sh
python3 deploy/build_router.py --source source-checkouts/llama-swap --output build/router
```

The build runs the upstream race tests and records the binary hash. Stock
llama-swap v250 cannot load a configuration using `ready`. Private guest ingress
and other operator-specific services are outside these model recipes.

For full GLM, additionally fetch `glm-full-base`, `glm-full-kernels`,
`glm-full-port`, and `glm-full-dcp` with `reproduce.py fetch`. Then use:

```sh
python3 deploy/build_images.py prepare glm-full --sources source-checkouts --output build/full
python3 deploy/build_images.py execute build/full/build-plan.json
```

This reconstructs the former local probe image from public eugr build source,
pinned vLLM/FlashInfer/DeepGEMM and CUDA inputs, disables automatic extra PRs,
then applies the kernels, DFlash2 port and SWA-under-MLA fix. The original build
did not record its CUDA base digest; this package records a newly resolved digest
for the same CUDA version. Build dependencies can also resolve differently.
The rebuilt image needs fresh acceptance; no bit-for-bit image claim is made.

## 2. Stage runtime and weights on each required node

Choose `glm-adaptive`, `qwen-single`, or `glm-full`. The paths must match the
site file. Run these as the service account on each node, before installation:

```sh
python3 deploy/stage.py runtime glm-adaptive --sources source-checkouts --output /srv/spark/runtime/glm-adaptive
python3 deploy/stage.py weights glm-adaptive --hf-home /srv/spark/huggingface --runtime /srv/spark/runtime/glm-adaptive
python3 deploy/stage.py weights glm-adaptive --hf-home /srv/spark/huggingface --runtime /srv/spark/runtime/glm-adaptive --execute
```

The first weight command previews downloads. The second downloads the exact
target and draft snapshots and, for GLM Flash, the pinned donor tensor ranges.
The transplant includes staged L15–L45; the DFlash2 target loader applies L15–L44.
The model is different if you omit it. The package carries no weights or access
tokens. Use the publisher's normal authentication if a snapshot requires it.

GLM Flash requires roughly 170 GB of target weights per rank plus draft, donor,
images and caches. Full GLM requires roughly 405 GB of target weights and a
4.92 GB draft per rank, plus substantial build and serving storage. A single
index/config file is not a complete checkpoint: staging checks every indexed shard.
For Qwen, the launcher requires the exact snapshot rather than choosing the
newest cached checkpoint.

## 3. Bind the site

Copy `examples/deployment-site.json` to a private location. Set the service user,
release path, shared state path, node SSH aliases, cache/runtime/model paths and
actual loaded image IDs. Set `image_references` for rebuilt images to the tags
in the build receipt. Paths must be absolute and contain no spaces or shell
metacharacters because the pinned upstream launchers construct remote commands.

The logical placement is fixed: pair A is `head` + `worker`; pair B is `rank3`
(head) + `rank2`. The mixed fleet places independent Qwen workers on rank2/rank3.
Full GLM uses rank order head → worker → rank2 → rank3 around the physical ring.
Configure the direct-link addresses, interfaces and RoCE GIDs from your hardware.
For full GLM, also configure the LAN interface/IP and both ring HCAs; the
switchless ring needs the matching subnet routes and 9000-byte MTU on its links.
Fabric provisioning is explicit host setup, not an inferred hostname convention.

```sh
PYTHONPATH=src python3 deploy/render_site.py --recipe glm53-flash-adaptive-4spark --site /private/path/site.json --output prepared/site
```

The output contains profiles, per-model adapters, the broker configuration and
the derived gateway configuration. It preserves context, batching, speculative
decoding and resource claims; site files cannot override those serving knobs.
`recipes prepare` is the older config-only helper; use the deployment renderer
for these portable adapters.

## 4. Install, then start through the broker

Copy the release, prepared bundle, built router binary and its adjacent JSON
receipt to every node. Use the same release path on every node:

```sh
sudo python3 deploy/install.py --bundle prepared/site --node head --router build/router/llama-swap-linux-arm64 --register
```

Use each node's logical ID in `--node`. Installation creates a new versioned
release, a Python environment, derived configuration, narrow resource helper
permissions and systemd units. It does not start models. The standard `/swap.img`
is supported for managed swap restoration; configure other swap separately.
Full GLM starts a root cache-flush service during its managed run, matching the
recorded memory setup, and stops it after unloading all ranks.

The installer refuses to overwrite a different existing service/helper. For an
existing fleet, omit `--register` to stage the release, then review the generated
`system/` files as part of a stopped-service migration. Preserve the old service,
helper, sudoers, site and gateway binary together for rollback. Re-render the
derived gateway JSON for both installation and rollback. Do not replace a live
gateway binary while expecting its profiles to remain unchanged.

On the head, once source, weights, images and fabric are ready:

```sh
sudo systemctl start dgx-spark-serve
/opt/spark-serve/releases/20260914/venv/bin/spark-serve --config /opt/spark-serve/releases/20260914/site/cluster.json status
```

Use `spark-serve serve MODEL` and `spark-serve stop MODEL` thereafter. Owned
container IDs are checked before removal, and launch refuses name/image drift.
The public API binds loopback; use an SSH tunnel or your chosen access layer.

## 5. Acceptance and recovery

Record the source/build receipts, loaded image IDs and site bundle hash. Check
API model identity, streamed text, a real tool call, and an image request for
Flash. Confirm context and batch geometry in the actual engine log. Exercise
managed stop, verify both/all ranks stop, then restart and repeat a request.
Do the stop/restart portion only after draining existing work.

```sh
PYTHONPATH=src python3 deploy/smoke.py --config /opt/spark-serve/releases/20260914/site/cluster.json --model glm-5.3-flash-exl3-ablit --output acceptance/api.json
PYTHONPATH=src python3 deploy/smoke.py --config /opt/spark-serve/releases/20260914/site/cluster.json --model glm-5.3-flash-exl3-ablit --output acceptance/restart.json --cycle
```

The second command deliberately reloads that model. It checks text, a tool call
and image understanding, verifies that all rank containers stopped, then repeats
the checks after the managed restart. Receipts record failures as failures.

With two GLM replicas, the native `local-auto--only--1` and `--2` profiles direct
new logical-route work to one ready replica without interrupting the other.
They do not cover direct physical-model callers. Account for those callers
before a reload. Full GLM occupies all four nodes and needs an all-fleet window.

Compare a small identical workload against the retained baseline after a rebuild.
Use completed output tokens per total wall time, with the same prompts and
concurrency. A whole benchmark campaign is needed only if those checks expose
a regression or the performance-relevant inputs change. Retain failed receipts;
do not label CPU preparation, a successful build, or a health response as full
deployment qualification.
