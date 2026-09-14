FROM dgx-local/glm53-exl3-e2:eb0469f-905c0293-c5d9c657
COPY exl3_fat_moe_ext.so /usr/local/lib/python3.12/dist-packages/exl3_fat_moe_ext.so
COPY test_e2_regression.py /opt/glm53/test_exl3_e2_regression.py
COPY selfcheck.py /opt/glm53/test_exl3_overlay.py
COPY e3-trial-check.py /opt/glm53/e3-trial-check.py
COPY exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
ENV EXL3_FAT_GROUPED=1 EXL3_FAT_EXPERT_LOG=1
LABEL dgx.trial="e3-only-20260909" dgx.source="dc6936cea8fd7b2e7ee5b7a48a5aa193857ca489" dgx.excludes="fp8-dense,adaptive,template-change"
RUN python3 -c "import torch, exl3_fat_moe_ext as m; from vllm.model_executor.layers.quantization import exl3; assert exl3.exl3_fat_moe_symbols(); assert exl3.grouped_fat_enabled(); assert exl3.fat_expert_log_enabled(); print('E3 symbols and matching diagnostics verified')"
