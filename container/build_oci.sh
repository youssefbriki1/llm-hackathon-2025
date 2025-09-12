#!/usr/bin/env bash

set -eEuo pipefail

BASE_IMAGE="docker.io/library/ubuntu:24.04"
OUTPUT_IMAGE_NAME="bigdft-hackathon"
OUTPUT_IMAGE_TAG="0.0.2"

container=$(buildah from "${BASE_IMAGE}")
echo "INFO: Started container from ${BASE_IMAGE}: ${container}"

echo "INFO: Installing dependencies..."
buildah run "${container}" -- apt-get update -y
buildah run "${container}" -- bash -c "DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install tzdata"
buildah run "${container}" -- apt-get update -y
# python3-venv for Yoann’s code.
buildah run "${container}" -- apt-get install -y --no-install-recommends python3 git sudo curl ca-certificates python3-venv rsync

echo "INFO: Configuring image..."

# Installing BigDFT
# First, because it uses its own Python.
buildah run "${container}" -- bash -c "curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh"
buildah config --env PATH=/opt/conda/bin:$PATH "${container}"
buildah run "${container}" -- /opt/conda/bin/conda init bash
buildah run "${container}" -- conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
buildah run "${container}" -- conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
buildah run "${container}" -- conda install -c conda-forge bigdft-suite -y

# Installing Aider
buildah config --env PATH=/root/.local/bin:$PATH "${container}"
buildah run "${container}" -- /opt/conda/bin/pip install aider-install
buildah run "${container}" -- /opt/conda/bin/aider-install

# Installing PyBigDFT
buildah run "${container}" -- git clone https://gitlab.com/luigigenovese/bigdft-suite.git
buildah config --workingdir /bigdft-suite "${container}"
buildah run "${container}" -- /opt/conda/bin/pip install -e PyBigDFT

# Installing hackathon and OntoFlow
buildah config --workingdir /work "${container}"
buildah run "${container}" -- git clone --recurse-submodules https://github.com/BigDFT-group/llm-hackathon-2025 /work/.
buildah run "${container}" -- /opt/conda/bin/pip install -r 2-aiengine/OntoFlow/agent/requirements.txt

# Installing BigDFT validator and runner
buildah run "${container}" -- /opt/conda/bin/pip install langgraph langgraph-supervisor langchain-openai langchain dotenv remotemanager

# Jupyter Lab for the interface.
buildah run "${container}" -- /opt/conda/bin/pip install jupyterlab

buildah config --workingdir /work "${container}"

echo "INFO: Committing the image..."

buildah commit --format oci --squash "${container}" "${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"
echo "INFO: Successfully committed OCI image: ${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"

buildah rm "${container}"
echo "INFO: Removed temporary container: ${container}"

echo "SUCCESS: OCI image built as ${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"
