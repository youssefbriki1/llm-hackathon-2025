#!/usr/bin/env bash

set -eEuo pipefail

BASE_IMAGE="docker.io/library/ubuntu:24.04"
OUTPUT_IMAGE_NAME="ontoflow"
OUTPUT_IMAGE_TAG="0.0.1"

container=$(buildah from "${BASE_IMAGE}")
echo "INFO: Started container from ${BASE_IMAGE}: ${container}"

echo "INFO: Installing dependencies..."
buildah run "${container}" -- apt-get update -y
buildah run "${container}" -- bash -c "DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install tzdata"
buildah run "${container}" -- apt-get update -y
buildah run "${container}" -- apt-get install -y --no-install-recommends python3 python3-pip python3-venv git

# Installing Aider
buildah config --env PATH=/root/.local/bin:$PATH "${container}"
buildah run "${container}" -- pip install --break-system-packages aider-install
buildah run "${container}" -- aider-install


echo "INFO: Configuring image..."

buildah config --workingdir /work "${container}"
buildah run "${container}" -- git clone https://github.com/Yopla38/OntoFlow /work/.

buildah run "${container}" -- pip install --break-system-packages -r agent/requirements.txt

echo "INFO: Committing the image..."

buildah commit --format oci --squash "${container}" "${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"
echo "INFO: Successfully committed OCI image: ${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"

buildah rm "${container}"
echo "INFO: Removed temporary container: ${container}"

echo "SUCCESS: OCI image built as ${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG}"
echo "To run a command with podman inside this image:"
echo "podman run --rm -it ${OUTPUT_IMAGE_NAME}:${OUTPUT_IMAGE_TAG} pwd"
