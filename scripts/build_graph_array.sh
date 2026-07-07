#!/bin/bash
#SBATCH --job-name=ctELM_build
#SBATCH --array=0-3
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_%A_%a.out
#SBATCH --error=logs/build_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1

VARIANTS=(
    configs/pipeline_no_labels_numbers.yaml
    configs/pipeline_no_labels_no_numbers.yaml
    configs/pipeline_labels_numbers.yaml
    configs/pipeline_labels_no_numbers.yaml
)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $VARIANT"

python -m openelm.graph --config configs/pipeline.yaml --variant "$VARIANT"
