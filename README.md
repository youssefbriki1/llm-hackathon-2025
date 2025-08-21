# llm-hackathon-2025

The purpose of this project is to explore how well some AI models can handle building new functionality on top of an existing electronic structure code. Our hypothesis is that a code with a well defined python API will be easier to work with than the current practice of CLI interactions. Our BigDFT code, with its associated PyBigDFT python bindings and remotemanager library for running on supercomputers, will be the underlying code.

## Raman Spectra
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
