#!/bin/bash
#SBATCH --job-name=ctELM_prepare_raw
#SBATCH --array=0-7
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/prepare_raw_%A_%a.out
#SBATCH --error=logs/prepare_raw_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
source "${SLURM_SUBMIT_DIR}/secrets.sh"

# Dataset rows (prompt_ids, domain_embedding_idx, target_idx) are representation-
# agnostic -- the same prepared dataset serves all 3 conditions (embedding,
# raw_finetune, raw_zeroshot), so this array is 8-wide by chain experiment only,
# on the base pipeline (no --variant). train_array_raw.sh/eval_array_raw.sh's
# 24-wide condition x experiment arrays all read from the output this writes.

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

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $EXPERIMENT"

python prepare_graph_dataset.py --config configs/pipeline.yaml --experiment "$EXPERIMENT"
