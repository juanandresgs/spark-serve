# Bounded prefill-row experiment

Apply `prefill-rows.patch` after the unchanged cancellation + burst patch parent.
The manifest binds its exact installed source hash. This independently written,
six-line change validates `TF_FLASH_PREFILL_ROWS` as `1024` or `2048` before GPU
allocation. When unset, the supplied constructor parameter remains untouched.
Both `self.prefill_rows` and the existing prefill buffer allocation use the same
selected value. There is no change to burst scheduling, cancellation, token
acceptance, sampler, prompt snapshots or FIFO packing code.

The experimental public derivative sets `TF_FLASH_PREFILL_ROWS=1024` while
retaining `TF_FLASH_DECODE_BURST=4`. Preserve the burst-only 2048-row baseline.
Smaller chunks may reduce decode stalls while increasing prefill overhead; no
performance claim follows from CPU checks.

Run `python -m unittest -v test_prefill_rows` in this directory against the patched
installed source. Expected: five tests, zero failures/errors/skips. Also rerun all
eight burst and six cancellation tests (19 total). The new tests exercise the
actual native constructor with only GPU allocation boundaries mocked and the
actual `_pieces` packing method, checking both bounds, partial FIFO prompts,
unset/custom parameter preservation, valid values, invalid values, and converged
buffer window sizing. GPU timing and quality qualification remain separate.
