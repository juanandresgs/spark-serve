ARG BASE
FROM ${BASE}
COPY overlay/patch_adaptive_k.py /opt/glm-throughput/overlay/patch_adaptive_k.py
COPY tests/test_adaptive_k_patch.py /opt/glm-throughput/tests/test_adaptive_k_patch.py
RUN python3 /opt/glm-throughput/tests/test_adaptive_k_patch.py && python3 /opt/glm-throughput/overlay/patch_adaptive_k.py
ENV GLM53_ADAPTIVE_K=ema GLM53_ADAPTIVE_K_SET=2,4,7 GLM53_ADAPTIVE_K_FILE=/root/.cache/vllm/glm-throughput-106.json
