# Project idea: LLM-powered HPC Research Assistant with remotemanager
We’re exploring a project on how LLMs can run scientific workflows on HPC systems, especially in the case when the underlying code is suitably (how?) documented and exposed via Python APIs.
The purpose of this project is to explore how well some AI models can handle building new functionality on top of an existing electronic structure code. Our hypothesis is that a code with a well defined python API will be easier to work with than the current practice of CLI interactions. Our BigDFT code, with its associated PyBigDFT python bindings and remotemanager library for running on supercomputers, will be the underlying code.

 Key tool: remotemanager (https://l_sim.gitlab.io/remotemanager/)— a Python library that submits and manages Python functions on remote supercomputers.
 Example code: (Py)BigDFT, the Python API of the BigDFT (bigdft.org) electronic-structure package, which serves as our test case for a DFT code with a Python interface.

## What we plan to do:
Use remotemanager to run tasks like atomisation energies or vibrational spectra on HPC.
Provide the agent with docs + tutorials written by developers, (e.g. BigDFT-school - https://github.com/BigDFT-group/bigdft-school).
Benchmark (how?) different models, study where documentation helps/fails, and identify ways to improve it (type hints, docstrings, examples).
Explore agentification: how to move from step-by-step prompting to autonomous agents orchestrating full workflows.

## Hackathon target: Build a prototype notebook assistant and derive guidelines for making scientific code “LLM-friendly.”
We’re therefore tryin to gather people interested in:

LLM prompting, RAG setups & agentification

Computational chemistry / materials science

HPC workflows & Python development

Creative presentation & visualization

# Present day plan about what to do

In our preliminary discussions we have pointed out that we may need to build a framework that would assist the user towards the creation of HPC-ready actions.
In order to do so we will likely gonna need a tool that, given a human-written documentation, reads that (either with RAG or keeping a large context) and try to answer to a user queries by producing some piece of code, that will have to be *validated* by some kind of agentic tool.
For some test problems, such "validator" should act as the same level of a CI of the assistan, that should be able to answer correctly to some pre-defined questions.

Asuggestion for our next steps (Giuseppe):

* Project title
* Definition of three/four validation tests which touch different points of the tutorials or bigdft capabilities (the validation tests have to include at least a fixed query and a known correct input file/response). I think in bigdft there is a biunivocal correspondence between the input file and the DFT results. In any case it is a sufficient condition.
* Definition of a protocol to submit a new development/test in our repository (I suggest well documented notebooks as we did with LangSim to help us internal communications, memory of previous work, writing of future reports/papers)
* Definition of the LLMs to be tested
* AI Approach: we have to use an agentic system, a RAG, a combination of agents and RAG, or something else?

# HCN test (Damien)

HCN molecule frequencies. This test comes in a descriptive way and it is on purpose chosen to be implementd with the old BigDFT API, such as to challenge the capability of the framework to capture new features from the documentation. See the `titan.pdf` file.


# Raman Spectra (potential future outcome)
The task will be to build a workflow for computing raman spectra of isolated molecules. Let's first though think of all the small steps that an LLM would need to accomplish to build this workflow.

First, it should be able to make the following plans:
1) Compute the phonon modes using a finite different approximation.
2) Compute the polarizability tensor using a series of finite field calculations.
3) Compute the raman spectra using the previous approaches.
4) Search the literature for reference data to compare against.

The core computational tasks are:
1) Run a single point energy calculation with BigDFT.
2) Determine the sensitivity of the energy to the grid spacing to understand a safe minimum finite difference step.
3) Run a single point energy calculation with BigDFT on a remote computer.
4) Run a single point 
5) Compute an optimized geometry with BigDFT.
6) Compute a single point energy after displacing an atom.
7) Compute the Hessian matrix and diagonalize it to get phonon modes.
8) Compute the polarizability tensor at displaced geometries along the phonon modes.
9) Calculate and plot raman intensities.
