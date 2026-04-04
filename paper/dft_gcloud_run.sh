#!/bin/bash
# =============================================================
# DFT Validation on Google Cloud C2-standard-16
#
# Run ALL 33 QE calculations on a 16-core VM
# Estimated time: 2-3 hours | Estimated cost: ~$2
#
# Usage:
#   1. Create VM:
#      gcloud compute instances create qe-dft \
#        --machine-type=c2-standard-16 \
#        --image-family=ubuntu-2204-lts \
#        --image-project=ubuntu-os-cloud \
#        --zone=us-central1-a \
#        --boot-disk-size=20GB
#
#   2. SSH in:
#      gcloud compute ssh qe-dft --zone=us-central1-a
#
#   3. Run this script:
#      bash dft_gcloud_run.sh
#
#   4. When done, copy results and DELETE the VM:
#      gcloud compute instances delete qe-dft --zone=us-central1-a
# =============================================================

set -e

NCORES=$(nproc)
WORK_DIR="$HOME/dft_validation"
REPO_DIR="$HOME/disorder-screening-agent"
INPUT_DIR="$REPO_DIR/dft_sqs_validation/qe_sqs_inputs"
PSEUDO_DIR="$REPO_DIR/dft_sqs_validation/pseudo"
RESULTS_DIR="$WORK_DIR/results"
CHECKPOINT_FILE="$RESULTS_DIR/checkpoint_all.json"

echo "============================================"
echo " DFT Validation — 33 calcs on $NCORES cores"
echo "============================================"

# ── Step 1: Install QE ──
echo ""
echo "[Step 1/4] Installing Quantum ESPRESSO..."
if ! command -v pw.x &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq quantum-espresso
    echo "  ✓ QE installed: $(pw.x --version 2>&1 | head -1)"
else
    echo "  ✓ QE already installed"
fi

# ── Step 2: Clone repo ──
echo ""
echo "[Step 2/4] Getting input files..."
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/snehalnair/disorder-screening-agent.git "$REPO_DIR"
    echo "  ✓ Repo cloned"
else
    echo "  ✓ Repo already exists"
fi

# ── Step 3: Set up work directory ──
echo ""
echo "[Step 3/4] Setting up work directory..."
mkdir -p "$WORK_DIR" "$RESULTS_DIR"

# Copy and patch input files
for f in "$INPUT_DIR"/*.in; do
    name=$(basename "$f")
    cp "$f" "$WORK_DIR/$name"

    # Patch parameters
    sed -i "s|pseudo_dir = './pseudo'|pseudo_dir = '$PSEUDO_DIR'|" "$WORK_DIR/$name"
    sed -i 's/ecutwfc = 40.0/ecutwfc = 60.0/' "$WORK_DIR/$name"
    sed -i 's/ecutrho = 320.0/ecutrho = 480.0/' "$WORK_DIR/$name"
    sed -i 's/mixing_beta = 0.1/mixing_beta = 0.35/' "$WORK_DIR/$name"
    sed -i 's/electron_maxstep = 500/electron_maxstep = 200/' "$WORK_DIR/$name"
    sed -i 's/tprnfor = .false./tprnfor = .true./' "$WORK_DIR/$name"
    sed -i 's/degauss = 0.02/degauss = 0.01/' "$WORK_DIR/$name"
done

echo "  ✓ $(ls "$WORK_DIR"/*.in | wc -l) input files patched"
echo "  ✓ ecutwfc=60, mixing_beta=0.35, tprnfor=.true."

# ── Step 4: Run all calculations ──
echo ""
echo "[Step 4/4] Running DFT calculations with $NCORES MPI ranks..."
echo ""

# Build list of all calculations
CALCS=(
    undoped_lith undoped_delith li_metal
    Al_ord_lith Al_ord_delith
    Al_sqs0_lith Al_sqs0_delith
    Al_sqs1_lith Al_sqs1_delith
    Ti_ord_lith Ti_ord_delith
    Ti_sqs0_lith Ti_sqs0_delith
    Ti_sqs1_lith Ti_sqs1_delith
    Mg_ord_lith Mg_ord_delith
    Mg_sqs0_lith Mg_sqs0_delith
    Mg_sqs1_lith Mg_sqs1_delith
    Ga_ord_lith Ga_ord_delith
    Ga_sqs0_lith Ga_sqs0_delith
    Ga_sqs1_lith Ga_sqs1_delith
    Fe_ord_lith Fe_ord_delith
    Fe_sqs0_lith Fe_sqs0_delith
    Fe_sqs1_lith Fe_sqs1_delith
)

TOTAL=${#CALCS[@]}
CONVERGED=0
FAILED=0
T_TOTAL_START=$(date +%s)

# Initialize checkpoint JSON
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Found existing checkpoint — will skip completed calcs"
else
    echo '{}' > "$CHECKPOINT_FILE"
fi

for i in "${!CALCS[@]}"; do
    name="${CALCS[$i]}"
    idx=$((i + 1))

    # Skip if already in checkpoint
    if grep -q "\"$name\"" "$CHECKPOINT_FILE" 2>/dev/null; then
        echo "[$idx/$TOTAL] $name — SKIPPED (in checkpoint)"
        continue
    fi

    echo -n "[$idx/$TOTAL] $name... "
    T_START=$(date +%s)

    inp_file="$WORK_DIR/${name}.in"
    out_file="$RESULTS_DIR/${name}.out"

    # Run QE with MPI
    mpirun -np $NCORES pw.x < "$inp_file" > "$out_file" 2>&1

    T_END=$(date +%s)
    ELAPSED=$((T_END - T_START))

    # Check convergence and extract energy
    if grep -q "convergence has been achieved" "$out_file"; then
        ENERGY_RY=$(grep '!' "$out_file" | tail -1 | awk -F'=' '{print $2}' | awk '{print $1}')
        ENERGY_EV=$(echo "$ENERGY_RY * 13.605698" | bc -l)
        TOTAL_FORCE=$(grep 'Total force' "$out_file" | tail -1 | awk '{print $4}')
        FORCE_EV=$(echo "$TOTAL_FORCE * 25.7112" | bc -l 2>/dev/null || echo "N/A")

        echo "✓ E=${ENERGY_RY} Ry (${ELAPSED}s, F=${FORCE_EV} eV/Å)"
        CONVERGED=$((CONVERGED + 1))

        # Update checkpoint JSON (append using python for proper JSON)
        python3 -c "
import json
with open('$CHECKPOINT_FILE') as f:
    d = json.load(f)
d['$name'] = {
    'energy_Ry': $ENERGY_RY,
    'energy_eV': $ENERGY_RY * 13.605698,
    'converged': True,
    'time_s': $ELAPSED,
    'total_force_Ry_bohr': float('$TOTAL_FORCE') if '$TOTAL_FORCE' != '' else None
}
with open('$CHECKPOINT_FILE', 'w') as f:
    json.dump(d, f, indent=2)
"
    else
        echo "✗ DID NOT CONVERGE (${ELAPSED}s)"
        FAILED=$((FAILED + 1))

        python3 -c "
import json
with open('$CHECKPOINT_FILE') as f:
    d = json.load(f)
d['$name'] = {'converged': False, 'time_s': $ELAPSED}
with open('$CHECKPOINT_FILE', 'w') as f:
    json.dump(d, f, indent=2)
"
    fi
done

T_TOTAL_END=$(date +%s)
T_TOTAL=$((T_TOTAL_END - T_TOTAL_START))

echo ""
echo "============================================"
echo " DONE: $CONVERGED/$TOTAL converged, $FAILED failed"
echo " Total time: $((T_TOTAL / 60)) min"
echo " Results: $RESULTS_DIR/"
echo " Checkpoint: $CHECKPOINT_FILE"
echo "============================================"

# ── Summary: compute voltages ──
echo ""
echo "Computing voltages..."
python3 << 'PYEOF'
import json

import os
ckpt = os.path.expanduser("~/dft_validation/results/checkpoint_all.json")
with open(ckpt) as f:
    results = json.load(f)

RY_TO_EV = 13.605698

# Get references
undoped_lith = results.get('undoped_lith', {})
undoped_delith = results.get('undoped_delith', {})
li_metal = results.get('li_metal', {})

if not all(r.get('converged') for r in [undoped_lith, undoped_delith, li_metal]):
    print("WARNING: Reference calcs did not all converge!")
else:
    E_li = li_metal['energy_eV'] / 2  # per atom (2-atom cell)

    # Undoped voltage
    V_undoped = -(undoped_delith['energy_eV'] - undoped_lith['energy_eV'] + 18 * E_li) / 18
    print(f"Undoped LiCoO2 voltage: {V_undoped:.3f} V (experiment: ~3.9 V)")
    print()

    # Dopant voltages
    dopants = ['Al', 'Ti', 'Mg', 'Ga', 'Fe']
    print(f"{'Dopant':>8s}  {'V_ord':>8s}  {'V_sqs0':>8s}  {'V_sqs1':>8s}  {'V_dis_mean':>10s}  {'|ord-dis|':>10s}")
    print("-" * 65)

    for dop in dopants:
        ord_l = results.get(f'{dop}_ord_lith', {})
        ord_d = results.get(f'{dop}_ord_delith', {})
        sqs0_l = results.get(f'{dop}_sqs0_lith', {})
        sqs0_d = results.get(f'{dop}_sqs0_delith', {})
        sqs1_l = results.get(f'{dop}_sqs1_lith', {})
        sqs1_d = results.get(f'{dop}_sqs1_delith', {})

        def voltage(lith, delith):
            if lith.get('converged') and delith.get('converged'):
                return -(delith['energy_eV'] - lith['energy_eV'] + 18 * E_li) / 18
            return None

        V_ord = voltage(ord_l, ord_d)
        V_sqs0 = voltage(sqs0_l, sqs0_d)
        V_sqs1 = voltage(sqs1_l, sqs1_d)

        V_dis_vals = [v for v in [V_sqs0, V_sqs1] if v is not None]
        V_dis_mean = sum(V_dis_vals) / len(V_dis_vals) if V_dis_vals else None

        diff = abs(V_ord - V_dis_mean) if V_ord and V_dis_mean else None

        print(f"{dop:>8s}  {V_ord:>8.3f}  {V_sqs0 or 'N/A':>8}  {V_sqs1 or 'N/A':>8}  {V_dis_mean or 'N/A':>10}  {diff or 'N/A':>10}")

    print()

    # Spearman rho
    from scipy.stats import spearmanr
    ord_voltages = []
    dis_voltages = []
    dop_names = []
    for dop in dopants:
        ord_l = results.get(f'{dop}_ord_lith', {})
        ord_d = results.get(f'{dop}_ord_delith', {})
        sqs0_l = results.get(f'{dop}_sqs0_lith', {})
        sqs0_d = results.get(f'{dop}_sqs0_delith', {})
        sqs1_l = results.get(f'{dop}_sqs1_lith', {})
        sqs1_d = results.get(f'{dop}_sqs1_delith', {})

        V_ord = voltage(ord_l, ord_d)
        V_sqs0 = voltage(sqs0_l, sqs0_d)
        V_sqs1 = voltage(sqs1_l, sqs1_d)
        V_dis_vals = [v for v in [V_sqs0, V_sqs1] if v is not None]
        V_dis_mean = sum(V_dis_vals) / len(V_dis_vals) if V_dis_vals else None

        if V_ord is not None and V_dis_mean is not None:
            ord_voltages.append(V_ord)
            dis_voltages.append(V_dis_mean)
            dop_names.append(dop)

    if len(ord_voltages) >= 3:
        rho, pval = spearmanr(ord_voltages, dis_voltages)
        print(f"DFT Spearman rho (ordered vs disordered): {rho:.3f} (p={pval:.4f}, n={len(ord_voltages)})")
        if rho < 0.85:
            print(">>> GATE G0 PASSES: Disorder changes DFT rankings!")
        else:
            print(">>> GATE G0 FAILS: Disorder does NOT change DFT rankings for this system")
    else:
        print(f"Not enough converged dopants for Spearman rho (n={len(ord_voltages)})")

PYEOF

echo ""
echo "============================================"
echo " NEXT STEPS:"
echo " 1. scp results to your local machine:"
echo "    gcloud compute scp qe-dft:~/dft_validation/results/* ./dft_results/ --zone=us-central1-a"
echo " 2. DELETE the VM to stop charges:"
echo "    gcloud compute instances delete qe-dft --zone=us-central1-a"
echo "============================================"
