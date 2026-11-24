#!/usr/bin/env bash
# Oblique flux-fix queue.  Run inside tmux on the remote box:
#
#     bash ~/DiffractML/v7-finite-grating/reports/m2/start_fix.sh
#
# Stages
#   0. Fresnel gate: pick the cell padding at 60 deg from a bare-interface
#      measurement (a wrong pad is what made the first 60 deg set unusable).
#   1. Repair the archived 30 deg references with the per-plane incident-flux
#      calibration (canary: exercises the fix on real data before the batch).
#   2. Regenerate 60 deg at the padding chosen by the gate (~5.5 days).
#   3. p-pol convergence + contrast ablation (MEEP side).
#
# Every stage is resumable: the generator skips existing JSONs, the repair
# skips references already present in data140_fix, and stage 2/3 skip their
# output files.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate meep-v7
cd ~/DiffractML/v7-finite-grating || exit 1

LOG=reports/m2/run_fix.log
JOBS=${JOBS:-2}
SPECS="${SPECS:-0 3 5 9 13 15 19 22}"
PY=python

mkdir -p reports/m2
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== start_fix: JOBS=$JOBS SPECS=$SPECS host=$(hostname) ==="

# ---------------------------------------------------------------- 0. gate --- #
if [ ! -f reports/m2/selfcheck.json ]; then
    log "stage 0: Fresnel self-check (bare interface vs analytic)"
    $PY scripts/flux_selfcheck.py --angles 30 60 --pads 1.5 3 4 \
        --pols s p --resolution 140 --out reports/m2/selfcheck.json \
        >>"$LOG" 2>&1
    log "stage 0 rc=$?"
else
    log "stage 0: selfcheck.json present, reusing"
fi

read_pad() {
    $PY - <<'PY'
import json
try:
    d = json.load(open("reports/m2/selfcheck.json"))
    print(d["chosen_pad"].get("60") or "")
except Exception:
    print("")
PY
}
PAD60=$(read_pad)
if [ -z "$PAD60" ]; then
    log "stage 0b: no pad <= 4 passed at 60 deg, trying pad=6"
    $PY scripts/flux_selfcheck.py --angles 60 --pads 6 --pols s p \
        --resolution 140 --out reports/m2/selfcheck.json --merge >>"$LOG" 2>&1
    PAD60=$(read_pad)
fi
if [ -z "$PAD60" ]; then
    log "GATE FAILED: 60 deg does not reproduce Fresnel at any pad; batch aborted"
    exit 1
fi
log "gate ok: pad(60 deg) = $PAD60"

# ------------------------------------------------------------- 1. repair --- #
NFIX=$(ls reports/m2/data140_fix/*_a30.json 2>/dev/null | wc -l)
if [ "$NFIX" -lt 16 ]; then
    log "stage 1: repairing archived 30 deg references ($NFIX/16 done)"
    ls reports/m2/data140/*_a30.json 2>/dev/null | sed 's#.*/##; s#\.json$##' \
        > reports/m2/repair_list.txt
    xargs -a reports/m2/repair_list.txt -d '\n' -P "$JOBS" -I B \
        bash -c 'python scripts/repair_reference_flux.py --angles 30 --only B \
                 >>reports/m2/run_fix.log 2>&1; \
                 echo "[repair] B rc=$?" >>reports/m2/run_fix.log'
    log "stage 1 done: $(ls reports/m2/data140_fix/*_a30.json 2>/dev/null | wc -l)/16 repaired"
else
    log "stage 1: already repaired ($NFIX/16)"
fi

# --------------------------------------------------------------- 2. 60 deg -- #
DEST="reports/m2/data140_pad${PAD60}"
mkdir -p "$DEST"
N60=$(ls "$DEST"/*.json 2>/dev/null | wc -l)
if [ "$N60" -lt 16 ]; then
    : > reports/m2/jobs_60.txt
    for i in $SPECS; do
        for pol in s p; do
            echo "$PY scripts/generate_m2_references.py --spec-set m2 --specs $i \
--pol $pol --angle 60 --n-freq 16 --resolution 140.0 --pad $PAD60 \
--out-dir $DEST" >> reports/m2/jobs_60.txt
        done
    done
    log "stage 2: $N60/16 present, launching $(wc -l < reports/m2/jobs_60.txt) jobs, pad=$PAD60"
    xargs -a reports/m2/jobs_60.txt -d '\n' -P "$JOBS" -I CMD \
        bash -c 'echo "[start] $(date +%H:%M:%S) CMD" >>reports/m2/run_fix.log; \
                 CMD >>reports/m2/run_fix.log 2>&1; \
                 echo "[done ] $(date +%H:%M:%S) CMD rc=$?" >>reports/m2/run_fix.log'
    log "stage 2 done: $(ls "$DEST"/*.json 2>/dev/null | wc -l)/16 references"
else
    log "stage 2: already complete ($N60/16)"
fi

# ------------------------------------------------------------ 3. p-pol ----- #
if [ ! -f reports/m2/pol_study/pol_study_meep.json ]; then
    log "stage 3: p-pol resolution sweep + contrast ablation"
    $PY scripts/pol_convergence.py --spec 3 --resolutions 140 200 280 \
        --skip-existing >>"$LOG" 2>&1
    log "stage 3 rc=$?"
else
    log "stage 3: already complete"
fi

log "=== start_fix finished ==="
