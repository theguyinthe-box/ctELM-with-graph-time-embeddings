import json
import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openelm.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Encode abstracts with a sentence transformer.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.variant, args.experiment)
    graph_outputd      = Path(cfg.paths.graph_outputd)
    embeddings_outputd = Path(cfg.paths.embeddings_outputd)
    embeddings_outputd.mkdir(parents=True, exist_ok=True)

    output_path = embeddings_outputd / "embeddings.npy"
    chkpt_path  = embeddings_outputd / "checkpoint"

    abstracts = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)
    model     = SentenceTransformer(cfg.embed_abstracts.model)
    dim       = cfg.embed_abstracts.embed_dim
    batch     = cfg.embed_abstracts.batch
    chkpt_every = cfg.embed_abstracts.chkpt

    start_idx = 0
    if chkpt_path.exists():
        with open(chkpt_path) as f:
            chkpt = json.load(f)
        start_idx = chkpt["last_idx"]
        emb = np.memmap(output_path, dtype="float32", mode="r+", shape=(len(abstracts), dim))
    else:
        emb = np.memmap(output_path, dtype="float32", mode="w+", shape=(len(abstracts), dim))

    for i in range(start_idx, len(abstracts), batch):
        vecs = model.encode(abstracts[i:i+batch], convert_to_numpy=True, show_progress_bar=False)
        emb[i:i+len(vecs)] = vecs
        if i > start_idx and (i // batch) % chkpt_every == 0:
            emb.flush()
            with open(chkpt_path, "w") as f:
                json.dump({"last_idx": i}, f)
            print(f"Embedded {i}/{len(abstracts)} abstracts")
    emb.flush()

if __name__ == "__main__":
    main()
