# DiffractML

***!! Still under research & development !!***

Modeling the optical response of diffractive structures using Rigorous
Coupled-Wave Analysis (RCWA) integrated with Machine Learning. The RCWA engine
is [TORCWA](https://github.com/Alexin-CH/TORCWA) (GPU-accelerated and
differentiable).

## Table of Contents

- [Active project](#active-project)
- [Archives](#archives)
- [Getting Started](#getting-started)

## Active project

- **`v6-inverse-design/`** — physics-informed inverse design of a 1D
  sine-corrugated TiN grating. A neural surrogate predicts the RCWA response,
  then inverse design recovers the grating geometry `(amp, per)` that matches a
  target response spectrum, with optional refinement through the differentiable
  RCWA solver. See its [README](v6-inverse-design/README.md).

## Archives

- **`archives/`** — earlier research versions (v0–v5): hyper-network weight
  prediction, order-convergence extrapolation, etc. Kept for reference; their
  data pipelines contained bugs (e.g. all 32 S-parameter columns identical) that
  are fixed in `v6-inverse-design`.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch (CUDA optional)
- Required libraries (listed in `requirements.txt`)

### Installation

Clone the repository (including submodules):

```bash
git clone --recurse-submodules https://github.com/Alexin-CH/DiffractML.git
cd DiffractML
```

Install the required dependencies:
``` bash
make
```

## Usage

See `v6-inverse-design/README.md` for the current project workflow
(generate → train → design). Regression tests:

```bash
cd v6-inverse-design && python -m pytest tests/
```