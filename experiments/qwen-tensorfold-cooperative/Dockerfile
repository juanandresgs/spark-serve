ARG BASE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE}
ARG TENSORFOLD_REV=34bae79ac97da6c3ab3fe10159cf49633ce8112a
RUN git init /opt/tensorfold-dependency && \
    git -C /opt/tensorfold-dependency remote add origin https://github.com/ashhart/TensorFold.git && \
    git -C /opt/tensorfold-dependency fetch --depth 1 origin ${TENSORFOLD_REV} && \
    git -C /opt/tensorfold-dependency checkout --detach FETCH_HEAD && \
    test "$(git -C /opt/tensorfold-dependency rev-parse HEAD)" = "${TENSORFOLD_REV}" && \
    python -m pip install --no-cache-dir /opt/tensorfold-dependency && \
    python -m pip freeze > /opt/tensorfold-dependencies.txt
LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold"
LABEL org.opencontainers.image.revision="34bae79ac97da6c3ab3fe10159cf49633ce8112a"

RUN python -m pip install --no-cache-dir jsonschema==4.26.0 && python -m pip freeze > /opt/tensorfold-dependencies.txt
COPY cooperative.py final_head.py profile_blocks.py tool_adapter.py telemetry.py serve.py /opt/spark-qwen/
ENV SPARK_COOPERATIVE=1 SPARK_PREFILL_ROWS=2048 SPARK_DECODE_BURST=4 SPARK_FINAL_HEAD=1 SPARK_PROFILE=0
ENTRYPOINT ["python", "/opt/spark-qwen/serve.py"]
