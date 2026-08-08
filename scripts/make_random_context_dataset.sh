#!/bin/bash
#SBATCH --job-name=ctELM_random_ctx
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/random_ctx_%j.out
#SBATCH --error=logs/random_ctx_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1

set -euo pipefail

VARIANT=${1:-configs/pipeline_no_labels_numbers.yaml}

python make_random_context_dataset.py --config configs/pipeline.yaml --variant "$VARIANT"

PREFIX=$(python -c "
from openelm.config import load_config
cfg = load_config('configs/pipeline.yaml', '$VARIANT')
print(cfg.paths.get('experiment_prefix', ''))
")

# symlink the already-trained chain2_generate checkpoint into
# chain2_random_context's output dir, so evaluate_model.py finds it there
# without retraining -- reusing chain2_generate's own output_dir would
# instead overwrite that run's eval_results.json/eval_predictions.jsonl.
CKPT_SRC_DIR="models/${PREFIX}/chain2_generate"
OUT_DIR="models/${PREFIX}/chain2_random_context"
mkdir -p "$OUT_DIR"
ln -sf "$(realpath "$CKPT_SRC_DIR"/checkpoint-*)" "$OUT_DIR"/
echo "Linked checkpoint from $CKPT_SRC_DIR into $OUT_DIR"
