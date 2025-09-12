# Deliverable

## Using the project

Get the OCI image from a repository, or build it with the script `build_oci.sh`.

You will need OpenAI and Anthropic keys for the setup to be fully functional:
create in this folder create files `anthropicAI_key.txt` and `openAI_key.txt` inside
a `keys` folder.

### Jupyter notebooks

Run the Jupyter from the container
```bash
podman run --rm -it -v $(pwd)/keys:/work/keys -p 8888:8888 -w /work bgidft-hackathon:0.0.2 \
  /opt/conda/bin/jupyter lab --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/work
```

Then open your browser to the URL indicated; you should be able to use the two notebooks.

### Using OntoFlow interractively

To test that everything is working, first make sure you have a `keys` folders with OpenAI
(for embeddings) and Anthropic keys (for structured responses) and run `podman`:
```bash
podman run --rm -it -v /path_to_keys_folder/keys:/work/keys ontoflow:0.0.1 bash
```

Then the script `Only_RAG.py` should work:
```bash
python3 -m Only_RAG
```

### Using a local GPU

The easiest way is to use Apptainer. We assume you have a SIF image. Then
```
apptainer run --containall --nv --writable-tmpfs ontoflow.sif bash
cd /work
```

## For developpers

To obtain latest version of the code and its dependencies:
```bash
git clone https://github.com/BigDFT-group/llm-hackathon-2025
cd llm-hackathon-2025
git submodule init
git submodule update --remote [--merge]
```
`--merge` at the end is if you want to update the main repository if there is a newer version of a submodule.
These commands should always be run from `llm-hackathon-2025`.
