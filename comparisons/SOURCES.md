# Public source recheck — September 30, 2026

The landing page preserves the historical figures in `public-references.json`.
They are attributed reference points, not matched experiments or independent
reproductions. Dates below distinguish the measurement from the source revision
and our recheck. No upstream implementation code was imported for this audit.

## TensorFold 0.6.0 evaluation lead — September 30

The [0.6.0 release notes](https://github.com/ashhart/TensorFold/releases/tag/v0.6.0)
describe CUDA prompt fill inside decode rounds, prompt resumption, consolidated
EXL3 tables and streamed typed tool arguments. These are upstream changes to
evaluate against the retained EXL3 pack. The release's explicit decoding and
prefill percentage gains concern NVFP4, so they are not evidence of an EXL3
speedup here. The current public experiment guide remains pinned to TensorFold
0.3.6.1. Any upgrade claim needs an identified same-pack control and candidate,
with C1/C4 throughput, quality, tools, cached resends and mixed traffic measured
separately.

## Qwen

[The pinned README at 856bb6b](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e/README.md)
and [the rechecked upstream README at a3aa898](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/README.md)
retain the same C4 prose and long-prompt prefill figures used in our table.
The table labels these as decode throughput and time to first token respectively.
Our local measurements count the whole request group or a completed JSON answer.
They cannot be used to calculate a cross-source speedup.

Public C4 and prefill observations used four streams; the current five-stream
default is a separate configuration. The public runtime is TensorFold 0.3.6.3,
with image/video support and different prefill chunks. Our recommended EXL3
recipe uses 0.3.6.1; the Affine4 alternative uses 0.3.6.2. Configured maximum context does not prove simultaneous
maximum-length prompt capacity.

## GLM

[The pinned README at dc6936c](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/dc6936cea8fd7b2e7ee5b7a48a5aa193857ca489/README.md)
contains two distinct historical workloads used in the public table:

- September 8: adaptive-k prose with FP8 dense layers, 850K context and a
  14 GiB KV cap. Our local recipe has BF16 dense layers and 11 GiB FP8 KV.
- August 28: high-acceptance structured/code decode, warm/empty KV, thinking off,
  temperature zero and a 400-token output setting. Its context and KV settings
  differ from the prose run as well as from our local code fixture.

[The rechecked upstream README at 674155d](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/674155dec2f2f62bb879801b5ce2cfc759a0bebf/README.md)
now reports September 28 default-versus-opt-in boost results from fresh boots,
with TTFT excluded from decode timing and FP8 dense layers by default. It also
reports a September 30 abliteration/boost comparison with one boot per arm.
These newer reports supersede the older figures as descriptions of that upstream
configuration. They are separately dated evidence, with different settings and
workloads, and do not supersede our original-site local receipts.

We preserve pinned history and link to the newer report rather than silently
replacing an unmatched reference with a new unmatched value. A current
implementation ranking requires identical prompts, sampling, output accounting,
cache state, precision and concurrency, plus repeated runs and quality checks.
The upstream-source audit is not a GPU replication. Fresh local EXL3 runs are recorded separately from these pinned public references.
