#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/ebekker/PlatonicTransformers/logs/%x_%j.out
#SBATCH --error=/home/ebekker/PlatonicTransformers/logs/%x_%j.err

# Usage: sbatch -J <job_name> qm9_regr_wd_sweep.sbatch <target> <weight_decay> [seed]
# Example: sbatch -J qm9-alpha-wd0p01 qm9_regr_wd_sweep.sbatch alpha 0.01
#          sbatch -J qm9-alpha-s1     qm9_regr_wd_sweep.sbatch alpha 1e-8 1

set -euo pipefail

TARGET="${1:?target (alpha, gap, ...) required}"
WD="${2:?weight_decay required}"
SEED="${3:-42}"

cd ~/PlatonicTransformers
source .venv/bin/activate
mkdir -p logs

echo "=== qm9_regr WD sweep on $(hostname) ==="
echo "target=$TARGET  weight_decay=$WD  seed=$SEED"
echo "branch: $(git rev-parse --abbrev-ref HEAD)  commit: $(git rev-parse --short HEAD)"
nvidia-smi -L

python mains/main_qm9_regr.py \
    --config configs/qm9_regr.yaml \
    --seed="$SEED" \
    --dataset.target="$TARGET" \
    --model.solid_name=tetrahedron \
    --model.compile=true \
    --model.attention_backend=flash \
    --optimizer.weight_decay="$WD" \
    --logging.wandb_identity=platonic-transformers-public
