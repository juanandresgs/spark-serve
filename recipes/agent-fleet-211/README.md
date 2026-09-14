# GLM coordinator and two independent Qwen workers

Historical status: **recommended-with-caveats**. Public profile templates have not been deployed or qualified.

| Model | Repository | Revision | Context |
|---|---|---|---:|
| glm-5.3-flash-exl3-ablit | Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw | `25a44fdbf16862a46b7cc9921142c6c81350af2f` | 262144 |
| qwen-worker-a | Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 | `925d7be6c14c6c9442ef83e8f05b5a3c39304f69` | 262144 |
| qwen-worker-b | Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 | `925d7be6c14c6c9442ef83e8f05b5a3c39304f69` | 262144 |

See [results](results.json), [source pins](../../sources.json), and [reproduction requirements](../../REPRODUCTION.md). Model identity includes any donor transplant described in each profile; substituting stock weights needs new evidence.

Profiles preserve logical placement, resource claims and recorded settings. UNBOUND paths require a site adapter and staged artifacts; preparation does not install them. This kit includes source preparation for E3 build inputs, not a complete fleet installer.
