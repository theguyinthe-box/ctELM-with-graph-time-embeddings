#!/bin/bash
#SBATCH --job-name=ctELM_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1

EXPERIMENT=${1:-}
VARIANT=${2:-}

ARGS=""
[ -n "$VARIANT" ]    && ARGS="$ARGS --variant $VARIANT"
[ -n "$EXPERIMENT" ] && ARGS="$ARGS --experiment $EXPERIMENT"

torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train.py --config configs/pipeline.yaml $ARGS
