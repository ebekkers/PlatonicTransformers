#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ebekker/PlatonicTransformers/logs/%x_%j.out
#SBATCH --error=/home/ebekker/PlatonicTransformers/logs/%x_%j.err

# Usage: sbatch -J <job_name> scripts/hipster/cifar10.sbatch <rope_on_values> [seed] [extra_cli]
# Example: sbatch -J cifar10-ls1e-4-s42 scripts/hipster/cifar10.sbatch true 42 \
#              "--model.freq_init=random --model.rope_v_separate_freqs=true --model.rope_sigma=32 --model.layer_scale_init_value=1e-4"

set -euo pipefail

ROPE_ON_VALUES="${1:?rope_on_values (true|false) required}"
SEED="${2:-42}"
EXTRA="${3:-}"

cd ~/PlatonicTransformers
source .venv/bin/activate
mkdir -p logs

export WANDB__SERVICE_WAIT=120

echo "=== cifar10 cyclic_4 attention on $(hostname) ==="
echo "rope_on_values=$ROPE_ON_VALUES  seed=$SEED  extra=$EXTRA"
echo "branch: $(git rev-parse --abbrev-ref HEAD)  commit: $(git rev-parse --short HEAD)"
nvidia-smi -L

python mains/main_cifar10.py \
    --config configs/cifar10_deit.yaml \
    --seed="$SEED" \
    --model.solid_name=cyclic_4 \
    --model.attention=true \
    --model.rope_on_values="$ROPE_ON_VALUES" \
    --dataset.data_dir=/home/ebekker/PlatonicTransformer/datasets/cifar10 \
    --system.num_workers=8 \
    --logging.wandb_identity=platonic-transformers-public \
    $EXTRA
