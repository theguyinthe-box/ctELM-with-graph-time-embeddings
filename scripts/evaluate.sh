#!/bin/bash
#SBATCH --job-name=ctELM_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
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

python evaluate_model.py --config configs/pipeline.yaml $ARGS
