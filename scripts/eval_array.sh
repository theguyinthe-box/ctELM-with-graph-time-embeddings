#!/bin/bash
#SBATCH --job-name=ctELM_eval
#SBATCH --array=0-31
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
source "${SLURM_SUBMIT_DIR}/secrets.sh"

VARIANTS=(
    configs/pipeline_no_labels_numbers.yaml
    configs/pipeline_no_labels_no_numbers.yaml
    configs/pipeline_labels_numbers.yaml
    configs/pipeline_labels_no_numbers.yaml
)

EXPERIMENTS=(
    configs/experiments/chain2_reconstruct.yaml
    configs/experiments/chain2_generate.yaml
    configs/experiments/chain3_reconstruct.yaml
    configs/experiments/chain3_generate.yaml
    configs/experiments/chain4_reconstruct.yaml
    configs/experiments/chain4_generate.yaml
    configs/experiments/chain5_reconstruct.yaml
    configs/experiments/chain5_generate.yaml
)

VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / 8))
EXP_IDX=$((SLURM_ARRAY_TASK_ID % 8))
VARIANT=${VARIANTS[$VARIANT_IDX]}
EXPERIMENT=${EXPERIMENTS[$EXP_IDX]}
echo "Task $SLURM_ARRAY_TASK_ID: $VARIANT + $EXPERIMENT"

python evaluate_model.py --config configs/pipeline.yaml --variant "$VARIANT" --experiment "$EXPERIMENT"
