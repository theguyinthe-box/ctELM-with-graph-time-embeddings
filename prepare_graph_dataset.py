import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, interleave_datasets, disable_progress_bars
from openelm.tokens_map import TOKEN_MAP_DICT
from openelm.graph.traverse import branch_iterator
from openelm.config import load_config

disable_progress_bars()

def graph_chain_generator(adj, abstracts, tokenizer, emb_token, gen_token, depth, n_total, seed, task=None):
    n_slots = task.prompt_template.count("{emb_token}") if task else 0
    include_target = task.get("include_target_embedding", True) if task else True
    # chain length = n_slots when target is included, n_slots+1 when withheld
    expected_chain_len = n_slots if include_target else n_slots + 1

    # the prompt is identical for every chain in this task (it only depends on
    # depth/task, not on which nodes end up in a given chain), so tokenize it
    # once here instead of re-running apply_chat_template on every row
    chain_len = depth + 1
    if n_slots > 0:
        prompt = task.prompt_template.format(emb_token=emb_token)
    else:
        indices_count = chain_len if include_target else chain_len - 1
        prompt = " ".join([emb_token] * indices_count)
    gen_tok_id = tokenizer.convert_tokens_to_ids(gen_token)
    probe_ids = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": gen_token},
    ])
    prompt_ids = probe_ids[:probe_ids.index(gen_tok_id) + 1]

    for chain in branch_iterator(adj, depth=depth, max_chains=n_total, seed=seed):
        if n_slots > 0 and len(chain) != expected_chain_len:
            continue
        if abstracts[chain[-1]] is None:
            continue
        indices = chain if include_target else chain[:-1]
        yield {
            "prompt_ids": prompt_ids,
            # store node indices and the target's node index, not embedding
            # values or tokenized target text — both are resolved from the
            # shared embeddings memmap / abstracts array at collate time, since
            # materializing them here duplicates the same data across every
            # overlapping chain and blows up dataset size combinatorially
            "domain_embedding_idx": [int(idx) for idx in indices],
            "target_idx": int(chain[-1]),
        }

def main():
    parser = argparse.ArgumentParser(description="Prepare graph chain dataset for ctELM.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config, args.variant, args.experiment)
    pcfg = cfg.prepare_graph_dataset

    if pcfg.base_model in TOKEN_MAP_DICT:
        token_map = TOKEN_MAP_DICT[pcfg.base_model]
    else:
        from transformers import AutoConfig
        from openelm.tokens_map import TYPE_TOKEN_MAP_DICT
        model_type = AutoConfig.from_pretrained(pcfg.base_model).model_type
        if model_type not in TYPE_TOKEN_MAP_DICT:
            raise ValueError(f"base_model '{pcfg.base_model}' has unsupported model_type '{model_type}'")
        token_map = TYPE_TOKEN_MAP_DICT[model_type]

    graph_shared       = Path(cfg.paths.graph_shared)
    graph_outputd      = Path(cfg.paths.graph_outputd)
    dataset_outputd    = graph_outputd / cfg.paths.dataset_subdir
    dataset_outputd.mkdir(parents=True, exist_ok=True)

    n_total = pcfg.n_train + pcfg.n_val + pcfg.n_eval

    print("Loading graph...")
    adj = sp.load_npz(graph_shared / "graph_adj.npz")

    print("Loading abstracts...")
    abstracts = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pcfg.base_model)
    emb_token = token_map["emb_tok"]
    gen_token  = token_map["gen_tok"]

    tasks     = pcfg.get("tasks", None)
    task_list = list(tasks) if tasks else [None]

    print(f"Sampling {n_total} chains per task ({len(task_list)} task(s))...")
    train_datasets = []
    val_datasets   = []
    eval_datasets  = []

    for task in task_list:
        ds = Dataset.from_generator(
            lambda t=task: graph_chain_generator(
                adj, abstracts, tokenizer,
                emb_token, gen_token, pcfg.depth, n_total, pcfg.seed, t
            )
        )
        ds = ds.shuffle(seed=pcfg.seed)
        n_available = len(ds)
        n_train, n_val = pcfg.n_train, pcfg.n_val
        if n_available < n_total:
            # fewer chains exist than requested (e.g. deep chains are rare) — scale
            # the split proportionally instead of crashing on an out-of-range index
            print(
                f"WARNING: only found {n_available:,} chains for task "
                f"'{task.task_name if task else None}', fewer than the requested "
                f"{n_total:,} (n_train={pcfg.n_train:,}, n_val={pcfg.n_val:,}, "
                f"n_eval={pcfg.n_eval:,}). Splitting proportionally instead."
            )
            n_train = int(pcfg.n_train / n_total * n_available)
            n_val   = int(pcfg.n_val   / n_total * n_available)
        train_datasets.append(ds.select(range(n_train)))
        val_datasets.append(  ds.select(range(n_train, n_train + n_val)))
        eval_datasets.append( ds.select(range(n_train + n_val, n_available)))

    def merge(dsets):
        return interleave_datasets(dsets) if len(dsets) > 1 else dsets[0]

    train_ds = merge(train_datasets)
    val_ds   = merge(val_datasets)
    eval_ds  = merge(eval_datasets)

    train_ds.save_to_disk(str(dataset_outputd / "train"))
    val_ds.save_to_disk(  str(dataset_outputd / "validation"))
    eval_ds.save_to_disk( str(dataset_outputd / "evaluation"))

    print(f"Done. {len(train_ds)} train / {len(val_ds)} validation / {len(eval_ds)} evaluation chains saved to {dataset_outputd}")

if __name__ == "__main__":
    main()
