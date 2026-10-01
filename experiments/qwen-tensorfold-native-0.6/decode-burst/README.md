# Native EXL3 decode-burst experiment

This independently written scheduling patch applies **after** the retained native
TensorFold v0.6.0 cancellation patch. It is an experimental derivative, not a
replacement for the cancellation-only baseline. No external recipe code is used.

`TF_FLASH_DECODE_BURST=1` is the default and preserves the original fill/decode
ordering. `TF_FLASH_DECODE_BURST=4` gives active decoders four completed decode
rounds between ordinary EXL3 prefill passes. Other values fail at construction,
before GPU buffer allocation. This is an explicit experimental setting; it does
not reinterpret the ineffective EXL3 `--decode-share` flag.

The first prefill pass still runs immediately. A decoder-only period resets that
priority for the next newly admitted prompt. A standalone fill never waits for a
decode budget. With active decoding and filling, each pass is followed by at most
four completed rounds before the next pass. The counter is updated only after
native decoding completes, not on scheduler polling or cancellation returns.
Cancellation remains the first operation of every round and is checked inside
prefill as before. Scheduler admission continues between rounds. The converged
(non-EXL3) mixed-forward branch is unchanged.

The patch does not change the 2048-row pass size, FIFO prompt packing, prompt
snapshots, KV cache, sampler, MTP depth, or token acceptance. Existing snapshot and
prompt completion boundaries remain entirely owned by unchanged `_pass`,
`_pieces`, `_absorb`, and `_joined` methods. A longer decode burst trades prompt
prefill completion latency for responsiveness of existing streams; only GPU
measurements establish whether that tradeoff is useful.

## Build and verify

Use a locally identified public cancellation-only image as the parent. Verify its
installed `multi.py` SHA-256 against `patch-manifest.json` before applying
`decode-burst.patch` with `patch -p1` from the installed source checkout. Verify
the resulting hash as well. No dependency update or network access is needed.
The source remains under the upstream license already carried by the public kit.

Run `python -m unittest -v test_decode_burst` with this directory on `PYTHONPATH`,
and rerun the existing `test_prefill_cancel` module from the cancellation package.
Expected: **8 scheduling tests + 6 cancellation tests, zero failures/errors/skips**.
The scheduling tests use the actual installed `MultiDecoder.round`, sampler,
acceptance and stream updates, mocking GPU staging/forward/state-buffer boundaries.
They cover default ordering, first pass, four completed rounds, bounded fill
progress, cancel while gated, decoder-only reset, standalone fill, multiple FIFO
fills, and valid/invalid settings. CPU checks do not prove GPU performance or cancellation
latency. Qualify burst 4 on a single released experimental host, preserving the
second host's cancellation-only baseline and production control.
