# DiffractML — setup and v7 experiment entry points.
#
# The project uses TWO conda environments (they cannot be merged):
#   rcwa-v7 : torch + TORCWA, for tests, training and every `make` target below
#   meep-v7 : pymeep, only for regenerating the FDTD references
#
# Typical first run:
#   make env-rcwa
#   conda activate rcwa-v7
#   make torcwa
#   make doctor
#   make test
#
# `make` uses the ACTIVE python. Override it when needed, e.g.
#   make test PYTHON=~/Apps/miniconda3/envs/rcwa-v7/bin/python

PYTHON    ?= python3
CONDA     ?= conda
V7         = v7-finite-grating

.DEFAULT_GOAL := help

.PHONY: help setup install env-rcwa env-meep torcwa doctor test \
        e3 e3-scale e4a e4b e4c manifest references dataset validate-oblique clean

help:
	@echo "DiffractML"
	@echo ""
	@echo "First install:"
	@echo "  make setup        print the install steps (does not run them)"
	@echo "  make env-rcwa     create the rcwa-v7 conda environment"
	@echo "  make env-meep     create the meep-v7 conda environment"
	@echo "  make torcwa       pip install -e TORCWA/ into the active env"
	@echo "  make doctor       verify the active interpreter"
	@echo ""
	@echo "Everyday use  [rcwa-v7]:"
	@echo "  make test         run the v7 test suite"
	@echo "  make e3           fair encoder ladder + sample-efficiency curve"
	@echo "  make e3-scale     extra seeds for the low-data regime"
	@echo "  make e4a          oracle effective periodic config"
	@echo "  make e4b          e4a + effective TiN permittivity"
	@echo "  make e4c          oracle low-order supercell at nh=8"
	@echo ""
	@echo "Ground truth  [meep-v7]:"
	@echo "  make manifest     emit the remote MEEP reference manifest"
	@echo "  make references   run the emitted manifest"
	@echo ""
	@echo "After the references land  [rcwa-v7]:"
	@echo "  make dataset      build the multi-(pol, angle) dataset"
	@echo "  make validate-oblique  check the 30 deg de-embedding vs MEEP"
	@echo ""
	@echo "  make clean        remove __pycache__ and pytest caches"
	@echo ""
	@echo "PYTHON=$(PYTHON)  (override with make <target> PYTHON=/path/to/python)"
	@echo "Inside an activated conda env, 'make test' uses that env's python3."

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
setup:
	@echo "1. conda env create -f $(V7)/environment-rcwa.yml"
	@echo "2. conda activate rcwa-v7"
	@echo "3. python -m pip install -e TORCWA/"
	@echo "4. conda env create -f $(V7)/environment.yml   # -> meep-v7, optional"
	@echo "5. make doctor && make test"
	@echo ""
	@echo "If 'conda' is not on PATH, source it first, e.g."
	@echo "  source .../miniconda3/etc/profile.d/conda.sh"

install: setup

env-rcwa:
	$(CONDA) env create -f $(V7)/environment-rcwa.yml

env-meep:
	$(CONDA) env create -f $(V7)/environment.yml

torcwa:
	$(PYTHON) -m pip install -e TORCWA/

doctor:
	@echo "[DOCTOR] Recap:"
	@$(PYTHON) -c "import sys, importlib.util as u; \
	mods = ['numpy', 'torch', 'torcwa', 'pytest', 'matplotlib']; \
	missing = [m for m in mods if u.find_spec(m) is None]; \
	print('python :', sys.executable); \
	print('pip    :', 'yes' if u.find_spec('pip') else 'MISSING -> conda install -n rcwa-v7 -c conda-forge pip'); \
	print('missing:', ', '.join(missing) if missing else 'none')"
	@if $(PYTHON) -c "import importlib.util as u,sys; sys.exit(0 if u.find_spec('meep') else 1)"; then \
		echo "meep   : available (this looks like meep-v7)"; \
	else \
		echo "meep   : not available in this venv (activate meep-v7 for 'make references')"; \
	fi
	@echo ""
	@echo "(rcwa-v7 needs numpy, torch, torcwa, pytest, matplotlib)"
	@echo "if anything is missing:  conda activate rcwa-v7 && make torcwa"

# --------------------------------------------------------------------------- #
# Tests  [rcwa-v7]
# --------------------------------------------------------------------------- #
test:
	cd $(V7) && $(PYTHON) -m pytest tests/ -q

# --------------------------------------------------------------------------- #
# Experiments  [rcwa-v7]
# --------------------------------------------------------------------------- #
E3_SPECS = --seeds 0 1 2 --n-train 6 12 18
E4_SPECS = 0 3 5 9 13 15 19 22
E4_MISSING = 0 5 13 22

e3:
	cd $(V7) && $(PYTHON) scripts/run_e3_suite.py --mode fair --mode scaling $(E3_SPECS)

e3-scale:
	cd $(V7) && $(PYTHON) scripts/run_e3_suite.py --mode scaling --n-train 6 --seeds 3 4 5

e4a:
	cd $(V7) && $(PYTHON) scripts/e4a_oracle_config.py --specs $(E4_SPECS) \
		--iters 14 --restarts 2 --n-fit-wl 5 --tag oracle_subset

e4b:
	cd $(V7) && $(PYTHON) scripts/e4a_oracle_config.py --specs $(E4_SPECS) \
		--iters 14 --restarts 2 --n-fit-wl 5 --eps-knob --tag oracle_eps

e4c:
	cd $(V7) && $(PYTHON) scripts/e4c_oracle_supercell.py --specs $(E4_MISSING) \
		--iters 12 --restarts 2 --tag oracle_missing

# --------------------------------------------------------------------------- #
# Remote MEEP references  [meep-v7]
# --------------------------------------------------------------------------- #
manifest:
	cd $(V7) && $(PYTHON) scripts/reference_manifest.py --sets A B C

references:
	cd $(V7) && bash reports/m2/run_parallel.sh

# --------------------------------------------------------------------------- #
# Dataset  [rcwa-v7]
# --------------------------------------------------------------------------- #
GEN_SPECS = 0 3 5 9 13 15 19 22

dataset:
	cd $(V7) && $(PYTHON) scripts/build_m2_dataset.py \
		--angles 0 30 60 --pols s p --specs $(GEN_SPECS) \
		--data-dir reports/m2/data140 \
		--out reports/m2/dataset_generalisation.npz

validate-oblique:
	cd $(V7) && $(PYTHON) scripts/validate_oblique.py \
		--spec 3 --pol s --angle 30 --data-dir reports/m2/data140

# --------------------------------------------------------------------------- #
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache $(V7)/.pytest_cache
