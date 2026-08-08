import json
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from pathlib import Path

# fitted gaussian over 2000 random embedding pairs, from semantic_similarity_distribution.py
NULL_MEAN, NULL_STD = 0.617, 0.047

VARIANT = "no_labels_numbers"
REAL_RESULTS   = Path(f"eval_results/{VARIANT}/chain2_generate/eval_results.json")
RANDOM_RESULTS = Path(f"eval_results/{VARIANT}/chain2_random_context/eval_results.json")


def load_cosine_sims(path):
    with open(path) as f:
        data = json.load(f)
    return np.array([row["cosine_similarity"] for row in data["per_example"]])


def main():
    real_sims   = load_cosine_sims(REAL_RESULTS)
    random_sims = load_cosine_sims(RANDOM_RESULTS)

    print(f"real chain context   (n={len(real_sims)}): mean={real_sims.mean():.4f} std={real_sims.std():.4f}")
    print(f"random context       (n={len(random_sims)}): mean={random_sims.mean():.4f} std={random_sims.std():.4f}")
    print(f"null baseline (unrelated embedding pairs):   mean={NULL_MEAN:.4f} std={NULL_STD:.4f}")

    fig = plt.figure()
    seaborn.histplot(real_sims, bins=60, stat="density", color="steelblue",
                      element="step", fill=False, label="real grandparent/parent context")
    seaborn.histplot(random_sims, bins=60, stat="density", color="crimson",
                      element="step", fill=False, label="random context (control)")
    plt.axvline(NULL_MEAN, color="gray", linestyle="--",
                label=f"null baseline ($\\mu$={NULL_MEAN:.3f})")
    plt.xlabel("cosine similarity (generated vs. real target)")
    plt.ylabel("density")
    plt.title(f"generated-vs-target cosine similarity, real vs. random context ({VARIANT})")
    plt.legend()
    out_path = "random_context_cosine_comparison.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
