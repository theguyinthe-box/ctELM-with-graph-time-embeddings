import argparse
import json
import os
import resource
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, Gemma3ForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import evaluate as hf_evaluate
from openelm.config import load_config
from openelm.model import LlamaForEmbeddingLM, Gemma3ForEmbeddingLM
from openelm.utils import expand_context_ids
from openelm.measure import (
    cuda_timer, memory_tracker, estimate_generate_flops, estimate_kv_cache_bytes,
    get_or_compute_abstract_token_lengths, compute_context_length_stats,
)
from openelm.tokens_map import TYPE_TOKEN_MAP_DICT

def resolve_context_mode(tcfg, ecfg):
    return ecfg.get("context_mode", tcfg.get("context_mode", "embedding"))

def load_model(tcfg, ecfg, output_dir=None):
    context_mode = resolve_context_mode(tcfg, ecfg)
    finetune     = tcfg.get("finetune", True)

    model_config = AutoConfig.from_pretrained(tcfg.basemodel_path)
    if context_mode == "raw_text":
        if model_config.model_type == "llama":
            model_class = LlamaForCausalLM
        elif model_config.model_type in ["gemma3", "gemma3_text"]:
            model_class = Gemma3ForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
    else:
        if model_config.model_type == "llama":
            model_class = LlamaForEmbeddingLM
        elif model_config.model_type in ["gemma3", "gemma3_text"]:
            model_class = Gemma3ForEmbeddingLM
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")

    base = model_class.from_pretrained(
        tcfg.basemodel_path,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()}
    )

    if finetune:
        checkpoint_dirs = sorted(
            Path(output_dir or tcfg.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoints found in {tcfg.output_dir}")
        checkpoint = str(checkpoint_dirs[-1])
        print(f"Loading checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
    else:
        print("finetune=false: evaluating the base model directly, no checkpoint loaded")
        model = base
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tcfg.basemodel_path)
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ctELM graph model.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config, args.variant, args.experiment)
    ecfg = cfg.eval
    tcfg = cfg.train

    prefix = getattr(cfg.paths, 'experiment_prefix', '')
    if prefix:
        p = Path(tcfg.output_dir)
        output_dir = str(p.parent / prefix / p.name)
    else:
        output_dir = tcfg.output_dir

    graph_outputd      = Path(cfg.paths.graph_outputd)
    embeddings_outputd = Path(cfg.paths.embeddings_outputd)
    dataset_outputd    = graph_outputd / cfg.paths.dataset_subdir

    context_mode = resolve_context_mode(tcfg, ecfg)
    lora_elm, tokenizer = load_model(tcfg, ecfg, output_dir=output_dir)
    model_config = AutoConfig.from_pretrained(tcfg.basemodel_path)
    # tokenizer.pad_token_id is unset on these base tokenizers; training uses
    # this model-specific id instead (see make_collate_function_dynamic_padding)
    pad_token_id = TYPE_TOKEN_MAP_DICT[model_config.model_type]["pad_tok_id"]

    abstracts  = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)
    embeddings = np.memmap(
        embeddings_outputd / "embeddings.npy",
        dtype="float32", mode="r", shape=(len(abstracts), cfg.embed_abstracts.embed_dim)
    )

    val_ds       = Dataset.load_from_disk(str(dataset_outputd / "evaluation"))
    embed_model  = SentenceTransformer(cfg.embed_abstracts.model)
    bertscore    = hf_evaluate.load(
        "bertscore",
        experiment_id=os.environ.get("SLURM_ARRAY_TASK_ID", str(os.getpid())),
    )

    context_length_stats = None
    if context_mode == "raw_text":
        max_seq_length = tcfg.get("max_seq_length", 2048)
        emb_tok_id = TYPE_TOKEN_MAP_DICT[model_config.model_type]["emb_tok_id"]
        sample_prompt_ids = val_ds[0]["prompt_ids"]
        prompt_overhead_tokens = len(sample_prompt_ids) - sample_prompt_ids.count(emb_tok_id)
        abstract_token_lens = get_or_compute_abstract_token_lengths(
            abstracts, tokenizer, cache_path=graph_outputd / f"abstract_token_lens_{model_config.model_type}.npy")
        context_length_stats = compute_context_length_stats(
            val_ds, abstract_token_lens, prompt_overhead_tokens, max_seq_length,
            include_target=True, target_token_lens=abstract_token_lens)
        print(f"Raw-text context length stats (evaluation split): {context_length_stats}")

    ctx_cache = {}
    metrics       = []   # per-example scalars only (no text) -- kept in memory for the full run
    batch_records = []
    batch_size    = ecfg.batch_size
    score_chunk_size = ecfg.get("score_chunk_size", 1000)

    predictions_path = Path(output_dir) / "eval_predictions.jsonl"
    pending = []   # rows awaiting a chunked bertscore pass: scalars + generated text
    row_id  = 0

    def flush_chunk(pending, pf):
        if not pending:
            return
        predictions = [p["generated"] for p in pending]
        references  = [str(abstracts[p["target_idx"]]) for p in pending]
        bs = bertscore.compute(predictions=predictions, references=references, lang="en")
        for p, prec, rec, f1 in zip(pending, bs["precision"], bs["recall"], bs["f1"]):
            metrics.append({
                "row_id":              p["row_id"],
                "target_idx":          p["target_idx"],
                "cosine_similarity":   p["cosine_similarity"],
                "bertscore_precision": prec,
                "bertscore_recall":    rec,
                "bertscore_f1":        f1,
                "context_tokens":      p["context_tokens"],
                "prompt_tokens":       p["prompt_tokens"],
                "generated_tokens":    p["generated_tokens"],
            })
            pf.write(json.dumps({"row_id": p["row_id"], "target_idx": p["target_idx"], "generated": p["generated"]}) + "\n")
        pending.clear()

    with open(predictions_path, "w") as pf:
        for batch_start in range(0, len(val_ds), batch_size):
            batch = val_ds[batch_start:batch_start + batch_size]

            # prompt_ids already ends with the gen token, so everything before that
            # last element is the prompt-only portion to feed into .generate()
            if context_mode == "raw_text":
                expanded = [
                    expand_context_ids(prompt_ids[:-1], context_idx, abstracts, tokenizer, emb_tok_id, ctx_cache)
                    for prompt_ids, context_idx in zip(batch["prompt_ids"], batch["domain_embedding_idx"])
                ]
                prompt_tensors      = [torch.tensor(ids, dtype=torch.long) for ids, _ in expanded]
                context_token_counts = [count for _, count in expanded]
            else:
                prompt_tensors = [
                    torch.tensor(prompt_ids[:-1], dtype=torch.long)
                    for prompt_ids in batch["prompt_ids"]
                ]
                context_token_counts = [len(idxs) for idxs in batch["domain_embedding_idx"]]

            # Left-pad so every row's real prompt ends at the same absolute index
            # (max_len) -- required for correct batched .generate(): the model
            # always continues generation from the last column of input_ids, so
            # right-padded shorter rows would otherwise continue from a pad token
            # instead of their real last prompt token.
            max_len = max(t.size(0) for t in prompt_tensors)
            padded         = torch.full((len(prompt_tensors), max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(prompt_tensors), max_len), dtype=torch.long)
            prompt_lengths = []
            for i, t in enumerate(prompt_tensors):
                padded[i, max_len - t.size(0):] = t
                attention_mask[i, max_len - t.size(0):] = 1
                prompt_lengths.append(t.size(0))
            padded         = padded.to("cuda")
            attention_mask = attention_mask.to("cuda")

            generate_kwargs = dict(
                input_ids=padded,
                attention_mask=attention_mask,
                max_new_tokens=ecfg.max_new_tokens,
                eos_token_id=lora_elm.config.eos_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=ecfg.repetition_penalty,
            )
            if context_mode != "raw_text":
                # Flatten domain_embedding_idx into resolved vectors: [ex0_emb0, ex0_emb1, ex1_emb0, ...]
                generate_kwargs["domain_embeddings"] = [
                    torch.tensor(embeddings[idx], dtype=torch.bfloat16).to("cuda")
                    for idxs in batch["domain_embedding_idx"]
                    for idx in idxs
                ]

            with memory_tracker() as mem, cuda_timer() as timer:
                with torch.no_grad():
                    outputs = lora_elm.generate(**generate_kwargs)

            # With left-padding, every row's prompt (real tokens + its own left
            # padding) occupies exactly max_len positions, so the generated
            # continuation starts at the same absolute index for every row.
            generated_texts = [
                tokenizer.decode(output[max_len:], skip_special_tokens=True)
                for output in outputs
            ]
            gen_embs = embed_model.encode(generated_texts, convert_to_numpy=True, show_progress_bar=False)

            batch_generated_tokens = 0
            for j, (output, prompt_len, generated) in enumerate(zip(outputs, prompt_lengths, generated_texts)):
                target_idx = batch["target_idx"][j]
                target_emb = np.array(embeddings[target_idx])
                cos_sim = float(np.dot(gen_embs[j], target_emb) / (np.linalg.norm(gen_embs[j]) * np.linalg.norm(target_emb) + 1e-8))

                # pad_token_id == eos_token_id for these tokenizers, so the first
                # occurrence after the prompt marks where real generation ended
                gen_slice     = output[max_len:]
                pad_positions = (gen_slice == pad_token_id).nonzero()
                generated_tokens = int(pad_positions[0].item()) + 1 if len(pad_positions) > 0 else int(gen_slice.numel())
                batch_generated_tokens += generated_tokens

                pending.append({
                    "row_id":            row_id,
                    "target_idx":        int(target_idx),
                    "generated":         generated,
                    "cosine_similarity": cos_sim,
                    "context_tokens":    context_token_counts[j],
                    "prompt_tokens":     prompt_len,
                    "generated_tokens":  generated_tokens,
                })
                row_id += 1

            flops   = estimate_generate_flops(model_config, prompt_len=max_len, n_generated_tokens=ecfg.max_new_tokens, batch_size=len(prompt_tensors))
            kv_bytes = estimate_kv_cache_bytes(model_config, seq_len=max_len + ecfg.max_new_tokens, batch_size=len(prompt_tensors))
            batch_records.append({
                "batch_start":                batch_start,
                "batch_size":                 len(prompt_tensors),
                "latency_s":                  timer["elapsed_s"],
                "throughput_examples_per_s":  len(prompt_tensors) / timer["elapsed_s"],
                "throughput_tokens_per_s":    batch_generated_tokens / timer["elapsed_s"],
                "peak_allocated_bytes":       mem["peak_allocated_bytes"],
                "peak_reserved_bytes":        mem["peak_reserved_bytes"],
                "prompt_tokens_max":          max_len,
                "est_flops_total":            flops["total_flops"],
                "est_kv_cache_bytes":         kv_bytes,
            })

            if len(pending) >= score_chunk_size:
                flush_chunk(pending, pf)

            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB on Linux
            print(f"  {min(batch_start + batch_size, len(val_ds))}/{len(val_ds)} examples evaluated (peak RSS: {rss_gb:.2f} GB)")

        print("Scoring final chunk...")
        flush_chunk(pending, pf)

    cos_sims = [m["cosine_similarity"] for m in metrics]
    bs_f1s   = [m["bertscore_f1"]      for m in metrics]
    summary  = {
        "n": len(metrics),
        "cosine_similarity": {"mean": float(np.mean(cos_sims)), "std": float(np.std(cos_sims))},
        "bertscore_f1":      {"mean": float(np.mean(bs_f1s)),   "std": float(np.std(bs_f1s))},
    }

    # token utilization / compute / memory instrumentation
    context_tokens_list   = [m["context_tokens"]   for m in metrics]
    prompt_tokens_list    = [m["prompt_tokens"]     for m in metrics]
    generated_tokens_list = [m["generated_tokens"]  for m in metrics]
    latencies             = [b["latency_s"]                 for b in batch_records]
    throughput_tokens     = [b["throughput_tokens_per_s"]   for b in batch_records]
    peak_allocs           = [b["peak_allocated_bytes"]      for b in batch_records]
    peak_reserved         = [b["peak_reserved_bytes"]       for b in batch_records]
    flops_totals          = [b["est_flops_total"]           for b in batch_records]
    kv_bytes_list         = [b["est_kv_cache_bytes"]         for b in batch_records]

    summary.update({
        "context_tokens":         {"mean": float(np.mean(context_tokens_list)),   "std": float(np.std(context_tokens_list))},
        "prompt_tokens":          {"mean": float(np.mean(prompt_tokens_list)),    "std": float(np.std(prompt_tokens_list))},
        "generated_tokens":       {"mean": float(np.mean(generated_tokens_list)), "std": float(np.std(generated_tokens_list))},
        "latency_s_per_batch":    {"mean": float(np.mean(latencies)),         "std": float(np.std(latencies))},
        "throughput_tokens_per_s":{"mean": float(np.mean(throughput_tokens)), "std": float(np.std(throughput_tokens))},
        "peak_allocated_bytes":   {"mean": float(np.mean(peak_allocs)),   "max": int(np.max(peak_allocs))},
        "peak_reserved_bytes":    {"mean": float(np.mean(peak_reserved)), "max": int(np.max(peak_reserved))},
        "est_flops_total":        {"mean": float(np.mean(flops_totals)), "sum": float(np.sum(flops_totals))},
        "est_kv_cache_bytes":     {"mean": float(np.mean(kv_bytes_list)), "max": int(np.max(kv_bytes_list))},
    })
    if context_length_stats is not None:
        summary["context_length_stats"] = context_length_stats

    # combined_score ranks examples for the best/worst generation sample below:
    # mean of min-max-normalized cosine similarity and BERTScore F1, so an example
    # has to do well on both the domain-embedding and text-overlap metrics to rank as "best".
    def min_max_normalize(values):
        values = np.array(values, dtype=float)
        lo, hi = values.min(), values.max()
        if hi - lo < 1e-12:
            return np.full_like(values, 0.5)
        return (values - lo) / (hi - lo)

    combined_scores = (min_max_normalize(cos_sims) + min_max_normalize(bs_f1s)) / 2
    for m, score in zip(metrics, combined_scores):
        m["combined_score"] = float(score)

    # Full generated/target text for every example would be too large to hold in
    # memory (n_eval is in the hundreds of thousands per experiment) so we only keep
    # the best and worst GENERATION_SAMPLE_FRACTION by combined_score for qualitative
    # review, re-reading their generated text from eval_predictions.jsonl below.
    GENERATION_SAMPLE_FRACTION = 0.05
    n_keep = min(max(1, int(len(metrics) * GENERATION_SAMPLE_FRACTION)), len(metrics) // 2)
    ranked = sorted(metrics, key=lambda m: m["combined_score"], reverse=True)
    best_worst_ids = {m["row_id"] for m in ranked[:n_keep]} | {m["row_id"] for m in ranked[-n_keep:]}

    text_by_row = {}
    with open(predictions_path) as pf:
        for line in pf:
            row = json.loads(line)
            if row["row_id"] in best_worst_ids:
                text_by_row[row["row_id"]] = row["generated"]

    def make_generation(m, group):
        return {
            "target_idx":     m["target_idx"],
            "quality_group":  group,
            "combined_score": m["combined_score"],
            "generated":      text_by_row[m["row_id"]],
            "target_text":    str(abstracts[m["target_idx"]]),
        }

    generations = (
        [make_generation(m, "best")  for m in ranked[:n_keep]]
        + [make_generation(m, "worst") for m in ranked[-n_keep:]]
    )

    # row_id is internal bookkeeping (disambiguates repeated target_idx values across
    # chains) -- not part of the saved per-example schema
    per_example = [{k: v for k, v in m.items() if k != "row_id"} for m in metrics]

    results_path     = Path(output_dir) / "eval_results.json"
    generations_path = Path(output_dir) / "eval_generations.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_example": per_example}, f, indent=2)
    with open(generations_path, "w") as f:
        json.dump(generations, f, indent=2)

    print(f"\n=== Evaluation Results ===")
    print(f"N:                  {summary['n']}")
    print(f"Cosine Similarity:  {summary['cosine_similarity']['mean']:.4f} ± {summary['cosine_similarity']['std']:.4f}")
    print(f"BERTScore F1:       {summary['bertscore_f1']['mean']:.4f} ± {summary['bertscore_f1']['std']:.4f}")
    print(f"Context tokens:     {summary['context_tokens']['mean']:.1f} ± {summary['context_tokens']['std']:.1f}")
    print(f"Throughput:         {summary['throughput_tokens_per_s']['mean']:.1f} tok/s")
    print(f"Peak GPU memory:    {summary['peak_allocated_bytes']['max'] / 1e9:.2f} GB")
    print(f"Saved metrics to {results_path}")
    print(f"Saved {len(generations)} generations ({n_keep} best + {n_keep} worst) to {generations_path}")
    print(f"Saved raw predictions to {predictions_path}")

if __name__ == "__main__":
    main()
