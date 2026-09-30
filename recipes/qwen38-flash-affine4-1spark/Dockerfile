ARG BASE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE}
ARG TF_REV=71377a5373ed7b394f1b480ba2a6a3986b03af1c
RUN git init /opt/tensorfold-dependency && git -C /opt/tensorfold-dependency remote add origin https://github.com/ashhart/TensorFold.git && git -C /opt/tensorfold-dependency fetch --depth 1 origin ${TF_REV} && git -C /opt/tensorfold-dependency checkout --detach FETCH_HEAD && test "$(git -C /opt/tensorfold-dependency rev-parse HEAD)" = "${TF_REV}" && python -m pip install --no-cache-dir /opt/tensorfold-dependency
RUN python -m pip install --no-cache-dir jsonschema==4.26.0 pybind11==3.0.4 && python -m pip freeze > /opt/tensorfold-dependencies.txt
COPY reader.cpp selection.py ssd_pipeline.py cooperative.py telemetry.py tool_adapter.py reliability.py serve.py /opt/spark-qwen/
RUN c++ -O3 -shared -std=c++17 -fPIC $(python -m pybind11 --includes) /opt/spark-qwen/reader.cpp -o /opt/spark-qwen/spark_positional_reader$(python3-config --extension-suffix) -pthread
ENV PYTHONPATH=/opt/spark-qwen SPARK_TILED_SELECT=1 SPARK_SSD_PIPELINE=1 SPARK_COOPERATIVE=1 SPARK_PREFILL_ROWS=2048 SPARK_DECODE_BURST=4 SPARK_FINAL_HEAD=0 SPARK_PROFILE=0
LABEL org.opencontainers.image.revision="71377a5373ed7b394f1b480ba2a6a3986b03af1c"
ENTRYPOINT ["python3", "/opt/spark-qwen/serve.py"]
