# Deliverable

## Using the project

Get the OCI (Docker) image from a registry, or build it with the script `build_oci.sh` using Buildah.

You will need OpenAI and Anthropic keys for the setup to be fully functional:
create files `anthropicAI_key.txt` and `openAI_key.txt` inside a `keys` folder.

### Jupyter notebooks

Run the Jupyter from the container. In two steps to have the correct environment.
```bash
docker run --rm -it -v $(pwd)/keys:/work/keys -p 8888:8888 -w /work bgidft-hackathon:0.0.2 bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/work
```
(If you want to use `podman` instead, just substitute `docker` with `podman`.)


You will have access to this repository in `/work`. You can then launch the two notebooks of this folder
* `ontoflow.ipynb` to have access to grounded code generation using RAG on documentation for PyBigDFT (OpenAI/Anthropic);
* `bigdft_validator.ipynb` to use the validator/executor of the generated code (OpenAI).
