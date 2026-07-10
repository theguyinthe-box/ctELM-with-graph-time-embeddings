import contextlib
import os
import time
import numpy as np
import torch

##########
## Timing / memory instrumentation
##########
@contextlib.contextmanager
def cuda_timer(device=None):
    """Wall-clock timer bracketed by torch.cuda.synchronize() so elapsed_s
    reflects actual GPU work, not just kernel-launch time. Do not nest two
    cuda_timer scopes on the same device -- each brackets a full sync."""
    result = {}
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    yield result
    torch.cuda.synchronize(device)
    result["elapsed_s"] = time.perf_counter() - start

@contextlib.contextmanager
def memory_tracker(device=None):
    """Peak CUDA memory over the wrapped block. Only one memory_tracker
    should be open on a given device at a time -- reset_peak_memory_stats
    is process/device-global, so a nested/concurrent tracker on the same
    device would corrupt the outer one's baseline."""
    result = {}
    torch.cuda.reset_peak_memory_stats(device)
    yield result
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

##########
## Analytic FLOPs / KV-cache estimates
##########
def estimate_causal_lm_params(model_config):
    """GQA-aware non-embedding param count from a HF PretrainedConfig.
    Excludes LoRA adapter and domain-embedding adapter weights (negligible
    relative to base-model size)."""
    hidden = model_config.hidden_size
    n_layers = model_config.num_hidden_layers
    n_heads = model_config.num_attention_heads
    n_kv_heads = getattr(model_config, "num_key_value_heads", n_heads)
    head_dim = getattr(model_config, "head_dim", hidden // n_heads)
    intermediate = model_config.intermediate_size
    vocab = model_config.vocab_size

    q_proj = hidden * (n_heads * head_dim)
    kv_proj = 2 * hidden * (n_kv_heads * head_dim)
    o_proj = (n_heads * head_dim) * hidden
    attn_params = q_proj + kv_proj + o_proj

    # SwiGLU-style MLP (gate + up + down), matches Llama/Gemma3 architecture
    mlp_params = 3 * hidden * intermediate

    params_per_layer = attn_params + mlp_params
    total_layer_params = params_per_layer * n_layers
    lm_head_params = hidden * vocab

    return {
        "params_per_layer": params_per_layer,
        "total_layer_params": total_layer_params,
        "lm_head_params": lm_head_params,
        "total_params": total_layer_params + lm_head_params,
    }

def estimate_generate_flops(model_config, prompt_len, n_generated_tokens, batch_size=1):
    """Analytic FLOPs for one generate() call: prefill over prompt_len
    tokens plus n_generated_tokens autoregressive decode steps, each
    attending over a growing KV cache. Excludes LoRA adapter and
    domain-embedding adapter compute (negligible: ~1e5-1e6 FLOPs/token vs
    ~1e10 base-model FLOPs/token)."""
    hidden = model_config.hidden_size
    n_layers = model_config.num_hidden_layers
    head_dim = getattr(model_config, "head_dim", hidden // model_config.num_attention_heads)
    n_heads = model_config.num_attention_heads

    params = estimate_causal_lm_params(model_config)
    # 2 FLOPs per param per token (multiply-add) for the linear/projection work
    linear_flops_per_token = 2 * params["total_layer_params"]
    lm_head_flops_per_token = 2 * params["lm_head_params"]

    def attention_flops(seq_len):
        # QK^T and softmax*V, both O(seq_len^2 * n_heads * head_dim), summed over layers
        return 2 * 2 * n_layers * n_heads * head_dim * (seq_len ** 2)

    prefill_flops = batch_size * (
        linear_flops_per_token * prompt_len
        + lm_head_flops_per_token * prompt_len
        + attention_flops(prompt_len)
    )

    decode_flops = 0
    for step in range(n_generated_tokens):
        ctx_len = prompt_len + step + 1
        decode_flops += batch_size * (
            linear_flops_per_token
            + lm_head_flops_per_token
            + 2 * 2 * n_layers * n_heads * head_dim * ctx_len
        )

    return {
        "prefill_flops": prefill_flops,
        "decode_flops": decode_flops,
        "total_flops": prefill_flops + decode_flops,
    }

def estimate_kv_cache_bytes(model_config, seq_len, batch_size=1, dtype_bytes=2):
    """2 (K&V) * num_layers * num_kv_heads * head_dim * seq_len * batch_size
    * dtype_bytes. Uses a uniform per-layer cache size for every layer; for
    Gemma3 (mixed sliding-window / full-attention layers, see
    model_config.layer_types/sliding_window) this overestimates the true
    cache since sliding-window layers cap their effective context at
    sliding_window tokens -- acceptable approximation for v1."""
    hidden = model_config.hidden_size
    n_layers = model_config.num_hidden_layers
    n_heads = model_config.num_attention_heads
    n_kv_heads = getattr(model_config, "num_key_value_heads", n_heads)
    head_dim = getattr(model_config, "head_dim", hidden // n_heads)

    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes

##########
## Token-utilization (context-length) measurement
##########
def get_or_compute_abstract_token_lengths(abstracts, tokenizer, cache_path):
    """Tokenizes every abstract once (add_special_tokens=False) and
    memoizes per-abstract token counts to cache_path (int32 .npy aligned to
    `abstracts`), loading from cache_path if it already exists. cache_path
    should be tokenizer-specific (e.g. include the model type in the
    filename) since token counts differ across tokenizers. Written via a
    temp file + os.replace for atomicity, so concurrent callers (e.g. a
    parallel SLURM sweep) racing to build the same cache don't corrupt it."""
    cache_path = str(cache_path)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    lengths = np.empty(len(abstracts), dtype=np.int32)
    for i, abstract in enumerate(abstracts):
        lengths[i] = len(tokenizer.encode(str(abstract), add_special_tokens=False)) if abstract is not None else 0

    tmp_path = f"{cache_path}.tmp-{os.getpid()}"
    np.save(tmp_path, lengths)
    # np.save appends .npy if the path doesn't already end with it
    tmp_path_actual = tmp_path if tmp_path.endswith(".npy") else tmp_path + ".npy"
    os.replace(tmp_path_actual, cache_path)
    return lengths

def compute_context_length_stats(
    dataset, abstract_token_lens, prompt_overhead_tokens, max_seq_length,
    include_target=False, target_token_lens=None, sample_size=20000, seed=42
):
    """Samples up to sample_size rows from `dataset` and sums
    abstract_token_lens over each row's domain_embedding_idx (plus the
    target's token length and prompt_overhead_tokens, if include_target),
    to estimate what full-length the row would be if fed as raw text.
    Returns {n_sampled, mean_tokens, p50, p90, p99, max_tokens,
    overflow_rate} where overflow_rate is the fraction exceeding
    max_seq_length."""
    n = len(dataset)
    if n == 0:
        return {"n_sampled": 0, "mean_tokens": 0, "p50": 0, "p90": 0, "p99": 0, "max_tokens": 0, "overflow_rate": 0.0}

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    rows = dataset[idx.tolist()]

    totals = np.empty(len(idx), dtype=np.int64)
    for i, (context_idx, target_idx) in enumerate(zip(rows["domain_embedding_idx"], rows["target_idx"])):
        total = int(sum(abstract_token_lens[j] for j in context_idx)) + prompt_overhead_tokens
        if include_target:
            total += int(target_token_lens[target_idx])
        totals[i] = total

    return {
        "n_sampled": int(len(totals)),
        "mean_tokens": float(np.mean(totals)),
        "p50": float(np.percentile(totals, 50)),
        "p90": float(np.percentile(totals, 90)),
        "p99": float(np.percentile(totals, 99)),
        "max_tokens": int(np.max(totals)),
        "overflow_rate": float(np.mean(totals > max_seq_length)),
    }

##########
## Training-time resource logging
##########
try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object

class ResourceLoggingCallback(TrainerCallback):
    """Captures step wall-time and peak GPU memory, merging them into the
    same `logs` dict HF's CallbackHandler passes through on_log -- lands in
    TrainingArguments.logging_dir at the existing logging_steps cadence,
    no new logging backend required."""

    def __init__(self):
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        self._step_start = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._step_start is None:
            return
        logs["step_latency_s"] = time.perf_counter() - self._step_start
        logs["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        logs["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
