#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ebekkers/PlatonicTransformers/logs/%x_%j.out
#SBATCH --error=/home/ebekkers/PlatonicTransformers/logs/%x_%j.err

# Usage: sbatch -J <job_name> scripts/snellius/cifar10.sbatch <rope_on_values> [seed]
# Example: sbatch -J cifar10-c4-rov0 scripts/snellius/cifar10.sbatch false
#          sbatch -J cifar10-c4-rov1 scripts/snellius/cifar10.sbatch true 42

set -euo pipefail

ROPE_ON_VALUES="${1:?rope_on_values (true|false) required}"
SEED="${2:-42}"

cd ~/PlatonicTransformers
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source /scratch-shared/ebekkers/platonic-pair-qm9/.venv/bin/activate

mkdir -p logs data

export WANDB__SERVICE_WAIT=120

echo "=== cifar10 cyclic_4 attention on $(hostname) ==="
echo "rope_on_values=$ROPE_ON_VALUES  seed=$SEED"
echo "branch: $(git rev-parse --abbrev-ref HEAD)  commit: $(git rev-parse --short HEAD)"
nvidia-smi -L

python mains/main_cifar10.py \
    --config configs/cifar10_deit.yaml \
    --seed="$SEED" \
    --model.solid_name=cyclic_4 \
    --model.attention=true \
    --model.rope_on_values="$ROPE_ON_VALUES" \
    --dataset.data_dir=/scratch-shared/ebekkers/cifar10 \
    --system.num_workers=8 \
    --logging.wandb_identity=platonic-transformers-public
