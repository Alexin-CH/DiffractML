# DiffractML — setup and v7 experiment entry points.
#
# Two environments are used (see v7-finite-grating/README.md):
#   meep-v7 : pymeep for the FDTD reference generation  (environment.yml)
#   rcwa-v7 : torch + TORCWA for RCWA and training       (environment-rcwa.yml)
#
# Override the interpreter when needed, e.g.
#   make test PYTHON=~/Apps/miniconda3/envs/rcwa-v7/bin/python

PYTHON    ?= python3
VENV_DIR  ?= venv
PIP        = $(VENV_DIR)/bin/pip
V7         = v7-finite-grating

.DEFAULT_GOAL := help

.PHONY: help all venv install env-meep env-rcwa torcwa test \
        e3 e3-scale e4a e4b e4c manifest references clean

help:
	@echo "DiffractML"
	@echo ""
	@echo "  make install      create ./$(VENV_DIR) and install requirements.txt"
	@echo "  make env-meep     create the MEEP conda environment (env-meep)"
	@echo "  make env-rcwa     create the RCWA/torch conda environment (env-rcwa)"
	@echo "  make torcwa       pip install -e TORCWA into \$$PYTHON"
	@echo "  make test         run the v7 test suite"
	@echo ""
	@echo "  make e3           fair encoder ladder + sample-efficiency curve"
	@echo "  make e3-scale     sample-efficiency curve only (6 seeds at n=6)"
	@echo "  make e4a          oracle effective periodic config (8-spec subset)"
	@echo "  make e4b          e4a + effective TiN permittivity"
	@echo "  make e4c          oracle low-order supercell at nh=8"
	@echo ""
	@echo "  make manifest     emit the remote MEEP reference manifest"
	@echo "  make references   run the emitted MEEP reference manifest"

all: install

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)

venv: $(VENV_DIR)

install: $(VENV_DIR)
	$(PIP) install -r requirements.txt

env-meep:
	conda env create -f $(V7)/environment.yml

env-rcwa:
	conda env create -f $(V7)/environment-rcwa.yml

torcwa:
	$(PYTHON) -m pip install -e TORCWA

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
test:
	cd $(V7) && $(PYTHON) -m pytest tests/ -q

# --------------------------------------------------------------------------- #
# Experiments (run in the rcwa-v7 environment)
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
# Remote MEEP references (run in the meep-v7 environment)
# --------------------------------------------------------------------------- #
manifest:
	cd $(V7) && $(PYTHON) scripts/reference_manifest.py --sets A B C

references:
	cd $(V7) && bash reports/m2/remote_manifest.sh

# --------------------------------------------------------------------------- #
clean:
	rm -rf $(VENV_DIR)
