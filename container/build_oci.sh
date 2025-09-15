#!/usr/bin/env bash

set -eEuo pipefail

BASE_IMAGE="docker.io/library/ubuntu:24.04"
OUTPUT_IMAGE_NAME="bigdft-hackathon"
OUTPUT_IMAGE_TAG="0.0.3"

container=$(buildah from "${BASE_IMAGE}")
echo "INFO: Started container from ${BASE_IMAGE}: ${container}"

echo "INFO: Installing dependencies..."
buildah run "${container}" -- bash -c ' \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt-get install -y --no-install-recommends \
        python3 \
        git \
        sudo \
        curl \
        ca-certificates \
        python3-venv \
        rsync \
        ssh && \
    rm -rf /var/lib/apt/lists/*'

echo "INFO: Configuring image..."

# Installing BigDFT
# First, because it uses its own Python.
buildah run "${container}" -- bash -c "curl -sSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm /tmp/mambaforge.sh"
buildah config --env PATH=/opt/conda/bin:$PATH "${container}"
buildah run "${container}" -- /opt/conda/bin/conda init bash
buildah run "${container}" -- bash -c "/opt/conda/bin/conda install -c conda-forge bigdft-suite -y && \
    /opt/conda/bin/conda clean -afy"

# Installing PyBigDFT
buildah run "${container}" -- bash -c ' \
    git clone --depth 1 https://gitlab.com/luigigenovese/bigdft-suite.git && \
    cd bigdft-suite && \
    /opt/conda/bin/pip install --no-cache-dir -e PyBigDFT && \
    rm -rf /bigdft-suite/.git'
buildah config --workingdir /bigdft-suite "${container}"

# Installing hackathon and OntoFlow
buildah config --workingdir /work "${container}"
buildah run "${container}" -- bash -c ' \
    git clone --depth 1 https://github.com/BigDFT-group/llm-hackathon-2025 /work/. && \
    cd /work && \
    git submodule update --init --recursive --depth 1 && \
    (cd 2-aiengine/OntoFlow && git fetch --depth 1 origin cpu && git checkout -b cpu FETCH_HEAD) && \
    /opt/conda/bin/pip install --no-cache-dir -r /work/2-aiengine/OntoFlow/agent/requirements.txt && \
    rm -rf /work/.git /work/2-aiengine/OntoFlow/.git'

buildah run "${container}" -- /opt/conda/bin/pip install --no-cache-dir \
    langgraph \
    langgraph-supervisor \
    langchain-openai \
    langchain \
    dotenv \
    remotemanager \
    jupyterlab

buildah config --workingdir /work "${container}"

echo "INFO: Committing the image..."

buildah commit --format oci --squash "${container}" "${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"
echo "INFO: Successfully committed OCI image: ${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"

buildah rm "${container}"
echo "INFO: Removed temporary container: ${container}"

echo "SUCCESS: OCI image built as ${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"
