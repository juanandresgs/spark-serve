ARG BASE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE}

ARG TENSORFOLD_REV=c4646171139ee8a3c38103eaa1699dad226ec12b
LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold" \
      org.opencontainers.image.revision="${TENSORFOLD_REV}"

COPY runtime-constraints.txt /opt/spark-serve/runtime-constraints.txt
COPY expected-base-inventory.json expected-primary-inventory.json inventory.py verify_runtime.py /opt/spark-serve/

# Resolve only against the exact package versions in the tested native image.
# The inventory check fails if dependency resolution changes the expected set.
RUN python /opt/spark-serve/inventory.py > /opt/spark-serve/base-inventory.json && \
    git init /opt/tensorfold-source && \
    git -C /opt/tensorfold-source remote add origin https://github.com/ashhart/TensorFold.git && \
    git -C /opt/tensorfold-source fetch --depth 1 origin "${TENSORFOLD_REV}" && \
    git -C /opt/tensorfold-source checkout --detach FETCH_HEAD && \
    test "$(git -C /opt/tensorfold-source rev-parse HEAD)" = "${TENSORFOLD_REV}" && \
    python -m pip install --no-cache-dir --constraint /opt/spark-serve/runtime-constraints.txt /opt/tensorfold-source && \
    python /opt/spark-serve/inventory.py > /opt/spark-serve/runtime-inventory.json && \
    python /opt/spark-serve/verify_runtime.py \
      --base /opt/spark-serve/base-inventory.json \
      --runtime /opt/spark-serve/runtime-inventory.json \
      --constraints /opt/spark-serve/runtime-constraints.txt \
      --expected /opt/spark-serve/expected-primary-inventory.json \
      --expected-base /opt/spark-serve/expected-base-inventory.json \
      --revision "${TENSORFOLD_REV}" \
      --check-dependencies \
      --output /opt/spark-serve/build-inventory-receipt.json && \
    python -c 'import importlib.metadata as m; assert m.version("tensorfold") == "0.6.0"'

ENTRYPOINT ["tensorfold"]
