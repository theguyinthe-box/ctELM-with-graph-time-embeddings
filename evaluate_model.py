import argparse
import json
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

    abstracts  = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)
    embeddings = np.memmap(
        embeddings_outputd / "embeddings.npy",
        dtype="float32", mode="r", shape=(len(abstracts), cfg.embed_abstracts.embed_dim)
    )

    val_ds       = Dataset.load_from_disk(str(dataset_outputd / "evaluation"))
    embed_model  = SentenceTransformer(cfg.embed_abstracts.model)
    bertscore    = hf_evaluate.load("bertscore")

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
    results       = []
    batch_records = []
    batch_size    = ecfg.batch_size

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

        max_len = max(t.size(0) for t in prompt_tensors)
        padded  = torch.full((len(prompt_tensors), max_len), tokenizer.pad_token_id, dtype=torch.long)
        prompt_lengths = []
        for i, t in enumerate(prompt_tensors):
            padded[i, :t.size(0)] = t
            prompt_lengths.append(t.size(0))
        padded = padded.to("cuda")

        generate_kwargs = dict(
            input_ids=padded,
            max_new_tokens=ecfg.max_new_tokens,
            eos_token_id=lora_elm.config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
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

        batch_generated_tokens = 0
        for j, (output, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
            generated   = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            target_idx  = batch["target_idx"][j]
            target_text = str(abstracts[target_idx])

            gen_emb    = embed_model.encode(generated, convert_to_numpy=True)
            target_emb = np.array(embeddings[target_idx])
            cos_sim    = float(np.dot(gen_emb, target_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(target_emb) + 1e-8))

            # pad_token_id == eos_token_id for these tokenizers, so the first
            # occurrence after the prompt marks where real generation ended
            gen_slice     = output[prompt_len:]
            pad_positions = (gen_slice == tokenizer.pad_token_id).nonzero()
            generated_tokens = int(pad_positions[0].item()) + 1 if len(pad_positions) > 0 else int(gen_slice.numel())
            batch_generated_tokens += generated_tokens

            results.append({
                "target_idx":        int(target_idx),
                "generated":         generated,
                "target_text":       target_text,
                "cosine_similarity": cos_sim,
                "context_tokens":    context_token_counts[j],
                "prompt_tokens":     prompt_len,
                "generated_tokens":  generated_tokens,
            })

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

        print(f"  {min(batch_start + batch_size, len(val_ds))}/{len(val_ds)} examples evaluated")

    # BERTScore computed in one pass over all results
    print("Computing BERTScore...")
    bs = bertscore.compute(
        predictions=[r["generated"]    for r in results],
        references= [r["target_text"]  for r in results],
        lang="en"
    )
    for i, r in enumerate(results):
        r["bertscore_precision"] = bs["precision"][i]
        r["bertscore_recall"]    = bs["recall"][i]
        r["bertscore_f1"]        = bs["f1"][i]

    cos_sims = [r["cosine_similarity"] for r in results]
    bs_f1s   = [r["bertscore_f1"]      for r in results]
    summary  = {
        "n": len(results),
        "cosine_similarity": {"mean": float(np.mean(cos_sims)), "std": float(np.std(cos_sims))},
        "bertscore_f1":      {"mean": float(np.mean(bs_f1s)),   "std": float(np.std(bs_f1s))},
    }

    # token utilization / compute / memory instrumentation
    context_tokens_list   = [r["context_tokens"]   for r in results]
    prompt_tokens_list    = [r["prompt_tokens"]     for r in results]
    generated_tokens_list = [r["generated_tokens"]  for r in results]
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
    for r, score in zip(results, combined_scores):
        r["combined_score"] = float(score)

    metrics = [
        {
            "target_idx":          r["target_idx"],
            "cosine_similarity":   r["cosine_similarity"],
            "bertscore_precision": r["bertscore_precision"],
            "bertscore_recall":    r["bertscore_recall"],
            "bertscore_f1":        r["bertscore_f1"],
            "combined_score":      r["combined_score"],
            "context_tokens":      r["context_tokens"],
            "prompt_tokens":       r["prompt_tokens"],
            "generated_tokens":    r["generated_tokens"],
        }
        for r in results
    ]

    # Full generated/target text for every example would be too large to store
    # (n_eval is in the hundreds of thousands per experiment) so we only keep the
    # best and worst GENERATION_SAMPLE_FRACTION by combined_score for qualitative review.
    GENERATION_SAMPLE_FRACTION = 0.05
    n_keep = min(max(1, int(len(results) * GENERATION_SAMPLE_FRACTION)), len(results) // 2)
    ranked = sorted(results, key=lambda r: r["combined_score"], reverse=True)
    best_worst = [(r, "best") for r in ranked[:n_keep]] + [(r, "worst") for r in ranked[-n_keep:]]

    generations = [
        {
            "target_idx":     r["target_idx"],
            "quality_group":  group,
            "combined_score": r["combined_score"],
            "generated":      r["generated"],
            "target_text":    r["target_text"],
        }
        for r, group in best_worst
    ]

    results_path     = Path(output_dir) / "eval_results.json"
    generations_path = Path(output_dir) / "eval_generations.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_example": metrics}, f, indent=2)
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

if __name__ == "__main__":
    main()
