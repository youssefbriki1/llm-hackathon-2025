= Deliverable

== Using the project

Get the OCI image from a repository, or build it with the script `build_oci.sh`.

=== Using OpenAI/Anthropic

To test that everything is working, first make sure you have a `keys` folders with OpenAI
(for embeddings) and Anthropic keys (for structured responses) and run `podman`:
```bash
podman run --rm -it -v /path_to_keys_folder/keys:/work/keys ontoflow:0.0.1 bash
```

Then the script `Only_RAG.py` should work:
```bash
python3 -m Only_RAG
```

=== Using a local GPU

The easiest way is to use Apptainer. We assume you have a SIF image. Then
```
apptainer run --containall --nv --writable-tmpfs ontoflow.sif bash
cd /work
```

== For developpers

To obtain latest version of the code and its dependencies:
```bash
git clone https://github.com/BigDFT-group/llm-hackathon-2025
git submodule init
git submodule update --remote --merge
```
