import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset, disable_progress_bars
from openelm.config import load_config

disable_progress_bars()

def main():
    parser = argparse.ArgumentParser(
        description="Build a random-context evaluation split: keeps the real "
                     "target but replaces the real grandparent/parent context "
                     "with 2 randomly sampled unrelated papers, as a control "
                     "for whether the model actually uses real chain context."
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--source-experiment", default="configs/experiments/chain2_generate.yaml")
    parser.add_argument("--out-subdir", default="dataset_chain2_random_context")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # source-experiment resolves paths.dataset_subdir to the real chain2_generate
    # evaluation split we're randomizing the context of
    cfg = load_config(args.config, args.variant, args.source_experiment)
    graph_outputd = Path(cfg.paths.graph_outputd)

    print("Loading abstracts...")
    abstracts = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)
    valid_idx = np.flatnonzero(np.array([a is not None for a in abstracts]))

    print("Loading source evaluation split...")
    src_ds = Dataset.load_from_disk(str(graph_outputd / cfg.paths.dataset_subdir / "evaluation"))

    rng = np.random.default_rng(args.seed)

    def randomize_context(row):
        exclude = {row["target_idx"], *row["domain_embedding_idx"]}
        picked = []
        while len(picked) < 2:
            candidate = int(rng.choice(valid_idx))
            if candidate not in exclude and candidate not in picked:
                picked.append(candidate)
        row["domain_embedding_idx"] = picked
        return row

    print(f"Randomizing context for {len(src_ds)} rows...")
    out_ds = src_ds.map(randomize_context)

    out_path = graph_outputd / args.out_subdir / "evaluation"
    out_ds.save_to_disk(str(out_path))
    print(f"Done. {len(out_ds)} random-context evaluation rows saved to {out_path}")

if __name__ == "__main__":
    main()
