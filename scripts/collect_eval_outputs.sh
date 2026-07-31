#!/bin/bash
# Run this ON MISHA (not locally). Mirrors just the eval output files
# (eval_results.json, eval_generations.json, eval_predictions.jsonl) out of
# models/${variant}/${experiment}/ into a flat staging dir, preserving the
# variant/experiment subpath but skipping every checkpoint-*/ directory --
# so the subsequent local `rsync` only has to walk a handful of small files
# instead of stat-ing every file under every checkpoint.
#
# find runs with -L (follow symlinks) since models/ (or output_dir) commonly
# points at project/scratch storage via a symlink -- without -L, find treats
# a symlinked top-level dir as an opaque file and silently descends into
# nothing.
set -euo pipefail

MODELS_DIR="${1:-models}"
STAGING_DIR="${2:-eval_exports}"

mkdir -p "$STAGING_DIR"

find -L "$MODELS_DIR" -type f \( \
    -name 'eval_results.json' -o \
    -name 'eval_generations.json' -o \
    -name 'eval_predictions.jsonl' \
  \) -print0 |
while IFS= read -r -d '' f; do
  rel="${f#"$MODELS_DIR"/}"
  dest="$STAGING_DIR/$rel"
  mkdir -p "$(dirname "$dest")"
  cp "$f" "$dest"
  echo "staged: $rel"
done

echo "Done. Pull it down from your local machine with:"
echo "  rsync -avzP misha.ycrc.yale.edu:~/project/ctELM-with-graph-time-embeddings/$STAGING_DIR/ ./$STAGING_DIR/"
