# Third-party provenance

The Spark Serve runtime and original project code retain their MIT license.
That license does not relicense model weights, upstream source, container
contents or derived patches. Source URLs and exact commits are in `sources.json`.

The pinned GLM E3 repository commit
`dc6936cea8fd7b2e7ee5b7a48a5aa193857ca489` contains GNU AGPL version 3 in its
`LICENSE` file. The exact license text is retained in
`licenses/GLM-E3-AGPL-3.0.txt`. The selected E3 integration/regression patches
and build inputs retain the upstream licensing scope and attribution to
MiaAI-Lab and the authors identified in that source. They are not represented
as MIT-only code. Fetching the source preserves its notices and source files.

The E2 source at `eb0469fbb2b49fd7c025f594a3339a121e58f7a9` was checked
and carries MIT, copyright 2026 Mia's AI Lab. Its exact notice is retained in
`licenses/GLM-E2-MIT.txt`.

The Qwen TP1 source at `78b0675e4469a19befaa9c84a41662b671d4185e` also
carries AGPL-3.0. Its exact text is retained in `licenses/Qwen-TP1-LICENSE.txt`;
the included Qwen loopback patch is not represented as MIT-only code.
The E2 integration and Qwen loopback patches modify their respective upstream
launchers. Preserve their license and author notices.
The full-model source references identify ajclark and tonyd2wild contributions.
Downloaded source and weights retain each publisher's terms.

The adaptive scheduler/graph patch derives from GLM Flash source at
`d5bf08a5f97062c3d1809bacc18af655ceec5120`, also AGPL-3.0. The deployment
candidate retains that license as `licenses/GLM-adaptive-LICENSE.txt`; the
patch and its corresponding public source are identified in `sources.json`.
The later upstream dense-precision and template changes are not adopted.

The full base builder at eugr revision `4ed3ebf71453cd3a19b5d3b49c8f841ca7d514bf`
is MIT licensed. The pinned ajclark DCP and tonyd2wild kernel repositories
carry Apache-2.0 notices. The pinned full-model port has no top-level LICENSE
file; this package fetches that source directly and does not redistribute a
copy of its repository. Keep its source headers and attributions with the
generated build inputs. The pinned llama-swap source is MIT licensed; its
source fetch retains `LICENSE.md` and the UI dependency notices.

This candidate does not bundle weights, containers, or the complete upstream
repositories. Before publishing derived image binaries, finish the per-image
source and notice inventory, including exact modifications and build inputs.
Binary redistribution and a complete license audit remain release gates.
