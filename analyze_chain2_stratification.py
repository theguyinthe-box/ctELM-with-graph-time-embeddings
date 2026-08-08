import json
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

VARIANT = "no_labels_numbers"
TASKS   = ["chain2_generate", "chain2_reconstruct"]
METRICS = ["cosine_similarity", "bertscore_f1", "combined_score"]


def load_rows(task):
    path = Path(f"eval_results/{VARIANT}/{task}/eval_results.json")
    with open(path) as f:
        data = json.load(f)
    rows = []
    for row in data["per_example"]:
        # chain_indices[0]/[1] are grandparent/parent in both generate (2-node)
        # and reconstruct (3-node, target appended) chain layouts
        gp_parent_cos = row["chain_cosine_similarities"][0]
        age_gap = row["chain_years"][1] - row["chain_years"][0]
        rows.append({
            "task": task,
            "gp_parent_cos": gp_parent_cos,
            "age_gap": age_gap,
            "cosine_similarity": row["cosine_similarity"],
            "bertscore_f1": row["bertscore_f1"],
            "combined_score": row["combined_score"],
        })
    return rows


def main():
    df = pd.DataFrame([r for task in TASKS for r in load_rows(task)])

    n_bad_gap = (df["age_gap"] < 0).sum()
    print(f"rows with negative age_gap (parent predates grandparent): {n_bad_gap} / {len(df)}")

    for task in TASKS:
        sub = df[df["task"] == task]
        print(f"\n=== {task} (n={len(sub)}) ===")
        for metric in METRICS:
            r_sim, p_sim = spearmanr(sub["gp_parent_cos"], sub[metric])
            r_gap, p_gap = spearmanr(sub["age_gap"], sub[metric])
            print(f"  {metric}: spearman(gp_parent_cos)={r_sim:+.3f} (p={p_sim:.2g})  "
                  f"spearman(age_gap)={r_gap:+.3f} (p={p_gap:.2g})")

        sub = sub.copy()
        sub["sim_tercile"] = pd.qcut(sub["gp_parent_cos"], 3, labels=["low", "mid", "high"])
        sub["gap_tercile"] = pd.qcut(sub["age_gap"].rank(method="first"), 3, labels=["short", "mid", "long"])
        pivot = sub.pivot_table(index="gap_tercile", columns="sim_tercile",
                                 values="combined_score", observed=False)

        fig = plt.figure()
        seaborn.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.xlabel("grandparent→parent cosine similarity")
        plt.ylabel("grandparent→parent age gap")
        plt.title(f"mean combined_score by similarity/age-gap tercile ({task}, {VARIANT})")
        out_path = f"chain2_stratification_{task}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"saved heatmap to {out_path}")


if __name__ == "__main__":
    main()
