#!/usr/bin/env bash
set -uo pipefail

# set B: 8 specs, s-pol, 0 deg
python scripts/generate_m2_references.py --spec-set m2 --specs 0 3 5 9 13 15 19 22 --pol s --angle 0 --n-freq 16 --resolution 140.0 --out-dir reports/m2/data140

# set B: 8 specs, s-pol, 30 deg
python scripts/generate_m2_references.py --spec-set m2 --specs 0 3 5 9 13 15 19 22 --pol s --angle 30 --n-freq 16 --resolution 140.0 --out-dir reports/m2/data140

# set B: 8 specs, s-pol, 60 deg
python scripts/generate_m2_references.py --spec-set m2 --specs 0 3 5 9 13 15 19 22 --pol s --angle 60 --n-freq 16 --resolution 140.0 --out-dir reports/m2/data140

# set B: 8 specs, p-pol, 0 deg
python scripts/generate_m2_references.py --spec-set m2 --specs 0 3 5 9 13 15 19 22 --pol p --angle 0 --n-freq 16 --resolution 140.0 --out-dir reports/m2/data140

# set B: 8 specs, p-pol, 30 deg
python scripts/generate_m2_references.py --spec-set m2 --specs 0 3 5 9 13 15 19 22 --pol p --angle 30 --n-freq 16 --resolution 140.0 --out-dir reports/m2/data140

# set B: 8 specs, p-pol, 60 deg
python scripts/generate_m2_references.py --spec-set m2 --specs 0 3 5 9 13 15 19 22 --pol p --angle 60 --n-freq 16 --resolution 140.0 --out-dir reports/m2/data140

