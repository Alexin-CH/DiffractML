#!/usr/bin/env bash
# Queue v2 for the oblique work.  Run inside tmux on the remote:
#
#     bash ~/DiffractML/v7-finite-grating/reports/m2/start_fix2.sh
#
# Stages (each resumable, in dependency order):
#   A1  repair the archived 30 deg references (per-plane flux fix)  ~5 h
#   A2  p-pol resolution sweep + constant-eps contrast ablation    ~4.5 h
#   B   lossless R+T gate for the oblique ratio correction          ~30 min
#   C0  bare-substrate controls for the 60 deg campaign cell        ~3 h
#   C1  16 MEEP references at 60 deg, one common cell               ~6 days
#   D   rebuild datasets and retrain (rcwa-v7)                      ~1 h
#
# The 60 deg campaign uses one common cell width so a single bare control per
# polarisation corrects every geometry (R = R0*R/R_bare).  The gate in B is a
# lossless structure, which must satisfy R+T=1: if it fails the campaign does
# not launch.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate meep-v7
# Two workers, two physical cores: pin every worker to a single thread or MEEP's
# FFT/OpenMP threads oversubscribe the box (load 4.5 on 4 cpus, ~5x slower).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd ~/DiffractML/v7-finite-grating || exit 1

LOG=reports/m2/run_fix2.log
JOBS=${JOBS:-2}
SPECS="${SPECS:-0 3 5 9 13 15 19 22}"
SX_C=${SX_C:-14.78}          # max aperture (7.78) + 2*(pad 3 + dpml 0.5)
PAD_C=${PAD_C:-3.0}
PY=python
mkdir -p reports/m2
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== start_fix2: JOBS=$JOBS SX_C=$SX_C host=$(hostname) ==="

# ------------------------------------------------------------------ A1 ----- #
NFIX=$(ls reports/m2/data140_fix/*_a30.json 2>/dev/null | wc -l)
if [ "$NFIX" -lt 16 ]; then
    log "A1: repairing 30 deg references ($NFIX/16)"
    ls reports/m2/data140/*_a30.json 2>/dev/null | sed 's#.*/##; s#\.json$##' \
        > reports/m2/repair_list.txt
    xargs -a reports/m2/repair_list.txt -d '\n' -P "$JOBS" -I B \
        bash -c 'python scripts/repair_reference_flux.py --angles 30 --only B \
                 >>reports/m2/run_fix2.log 2>&1; echo "[repair] B rc=$?" \
                 >>reports/m2/run_fix2.log'
    log "A1 done: $(ls reports/m2/data140_fix/*_a30.json 2>/dev/null | wc -l)/16"
else
    log "A1: already repaired ($NFIX/16)"
fi

# ------------------------------------------------------------------ A2 ----- #
if [ ! -f reports/m2/pol_study/pol_study_meep.json ]; then
    log "A2: p-pol resolution sweep + contrast ablation"
    $PY scripts/pol_convergence.py --spec 3 --resolutions 140 200 280 \
        --skip-existing >>"$LOG" 2>&1
    log "A2 rc=$?"
else
    log "A2: already complete"
fi

# ------------------------------------------------------------------ B ------ #
GATE=reports/m2/ratio_gate.json
if ! grep -q '"pass": true' "$GATE" 2>/dev/null; then
    log "B: lossless R+T gate at 60 deg (sx=$SX_C)"
    $PY scripts/validate_ratio_scheme.py --angle 60 --sx "$SX_C" \
        --resolution 100 --wl 0.75 0.90 1.05 >>"$LOG" 2>&1
    log "B rc=$?"
fi
if ! grep -q '"pass": true' "$GATE" 2>/dev/null; then
    log "B: GATE FAILED - the ratio correction does not restore R+T=1;"
    log "B: not starting the 6-day campaign. See $GATE"
    exit 1
fi
log "B: gate passed"

# ------------------------------------------------------------------ C0 ----- #
for pol in s p; do
    CTRL="reports/m2/bare/bare_a60_${pol}_sx${SX_C}.json"
    if [ ! -f "$CTRL" ]; then
        log "C0: bare control pol=$pol sx=$SX_C"
        $PY scripts/make_bare_control.py --angle 60 --pol "$pol" --sx "$SX_C" \
            --pad "$PAD_C" --n-freq 6 --resolution 140 >>"$LOG" 2>&1
        log "C0: $pol rc=$?"
    fi
done

# ------------------------------------------------------------------ C1 ----- #
DEST="reports/m2/data140_pad3c"
mkdir -p "$DEST"
N60=$(ls "$DEST"/*.json 2>/dev/null | wc -l)
if [ "$N60" -lt 16 ]; then
    : > reports/m2/jobs_60c.txt
    for i in $SPECS; do
        for pol in s p; do
            echo "$PY scripts/generate_m2_references.py --spec-set m2 --specs $i \
--pol $pol --angle 60 --n-freq 16 --resolution 140.0 --common-sx $SX_C \
--pad $PAD_C --energy-tol 0.15 --out-dir $DEST" >> reports/m2/jobs_60c.txt
        done
    done
    log "C1: $N60/16 present, launching $(wc -l < reports/m2/jobs_60c.txt) jobs"
    xargs -a reports/m2/jobs_60c.txt -d '\n' -P "$JOBS" -I CMD \
        bash -c 'echo "[start] $(date +%H:%M:%S) CMD" >>reports/m2/run_fix2.log; \
                 CMD >>reports/m2/run_fix2.log 2>&1; \
                 echo "[done ] $(date +%H:%M:%S) CMD rc=$?" >>reports/m2/run_fix2.log'
    log "C1 done: $(ls "$DEST"/*.json 2>/dev/null | wc -l)/16"
else
    log "C1: already complete ($N60/16)"
fi

# ------------------------------------------------------------------ D ------ #
conda activate rcwa-v7 || { log "D: rcwa-v7 missing"; exit 1; }
cp -n "$DEST"/* reports/m2/data140_fix/ 2>/dev/null
log "D: building datasets (rcwa-v7) and retraining"
$PY scripts/build_m2_dataset.py --angles 0 30 --pols s p --specs $SPECS \
    --data-dir reports/m2/data140_fix \
    --out reports/m2/dataset_gen_fix.npz >>"$LOG" 2>&1
log "D: 0/30 dataset rc=$?"
$PY scripts/build_m2_dataset.py --angles 0 30 60 --pols s p --specs $SPECS \
    --data-dir reports/m2/data140_fix --bare-dir reports/m2/bare \
    --out reports/m2/dataset_gen60.npz >>"$LOG" 2>&1
log "D: 0/30/60 dataset rc=$?"
$PY scripts/train_m2.py --data reports/m2/dataset_gen60.npz --epochs 800 \
    --hidden 128 --depth 4 --weight-decay 1e-3 --patience 150 --device cuda \
    --tag gen60 --out-dir reports/m2/gen60 >>"$LOG" 2>&1
log "D: training rc=$?"
log "=== start_fix2 finished ==="
