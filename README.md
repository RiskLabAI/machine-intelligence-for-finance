# Machine Intelligence for Finance

Executable notebooks and supporting code for *Machine Intelligence for Finance*.

The notebooks are arranged by chapter and section. Every Python notebook has a Julia counterpart under `notebooks/julia`. This repository contains code and notebook results only. It does not contain the manuscript, LaTeX sources, or a PDF of the text.

The Python examples use RiskLabAI 3.2.0. The Julia examples use RiskLabAI.jl 1.1.0. Both use public online data and visible fixed-seed simulations when a known population or counterfactual is required. They are teaching and research examples, not evidence of market performance.

## Running the Python notebooks

The notebooks retrieve the public files they need when they first run. The source list and local paths are recorded in `DATA_SOURCES.md`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
jupyter lab
```

Generated figures, tables, and intermediate files are written under `outputs/`. Notebook figures use a dark background and a shared color palette in both languages.

## Running the Julia notebooks

Create the Julia environment and start Jupyter from the repository root:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using IJulia; notebook(dir=".")'
```

Julia notebooks use the same public sources and fixed simulation designs. Prepare their local, ignored data cache with:

```powershell
python notebooks/julia/prepare_inputs.py
```

This command downloads the listed public files and recreates simulated inputs from the stated seeds. Neither downloaded data nor generated simulation files are committed to the repository.

The appendix notebook that compares the Python and Julia packages also requires a public RiskLabAI.jl 1.1.0 checkout. Set `RISKLABAI_JULIA_ROOT` to that checkout before running the comparison.

## Contents

See `NOTEBOOK_CROSSWALK.md` for the complete chapter list and notebook counts.

## Related material

Reusable method tutorials that are not tied to book order remain in the RiskLabAI Notebooks.py collection.

## License

This repository is released under the BSD 3-Clause License. See `LICENSE` for the complete terms.
