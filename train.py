import os
import json
import argparse
from pathlib import Path
import numpy as np
from openelm.model import LlamaForEmbeddingLM, Gemma3ForEmbeddingLM
from openelm.utils import (
    make_collate_function_dynamic_padding_llama, make_collate_function_dynamic_padding_gemma3,
    make_collate_function_raw_text_dynamic_padding_llama, make_collate_function_raw_text_dynamic_padding_gemma3,
)
from openelm.config import load_config
from openelm.measure import ResourceLoggingCallback, get_or_compute_abstract_token_lengths, compute_context_length_stats
from openelm.tokens_map import TYPE_TOKEN_MAP_DICT
from datasets import Dataset
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, LlamaForCausalLM, Gemma3ForCausalLM
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from torch.distributed.elastic.multiprocessing.errors import record
import torch

@record
def main():
    parser = argparse.ArgumentParser(description="Train a embedding language model.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config, args.variant, args.experiment)
    tcfg = cfg.train

    prefix = getattr(cfg.paths, 'experiment_prefix', '')
    if prefix:
        p = Path(tcfg.output_dir)
        output_dir = str(p.parent / prefix / p.name)
    else:
        output_dir = tcfg.output_dir

    graph_outputd      = Path(cfg.paths.graph_outputd)
    embeddings_outputd = Path(cfg.paths.embeddings_outputd)
    dataset_dir = graph_outputd / cfg.paths.dataset_subdir
    training_dataset = Dataset.load_from_disk(str(dataset_dir / "train"))
    dev_dataset      = Dataset.load_from_disk(str(dataset_dir / "validation"))

    abstracts  = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)
    embeddings = np.memmap(
        embeddings_outputd / "embeddings.npy",
        dtype="float32", mode="r", shape=(len(abstracts), cfg.embed_abstracts.embed_dim)
    )

    tokenizer = AutoTokenizer.from_pretrained(tcfg.basemodel_path)

    context_mode = tcfg.get("context_mode", "embedding")
    max_seq_length = tcfg.get("max_seq_length", 2048)

    config = AutoConfig.from_pretrained(tcfg.basemodel_path)
    if context_mode == "raw_text":
        if config.model_type == "llama":
            model_class = LlamaForCausalLM
            collate_fn  = make_collate_function_raw_text_dynamic_padding_llama(
                abstracts, tokenizer, max_seq_length=max_seq_length, on_overflow=tcfg.get("on_context_overflow", "warn"))
        elif config.model_type in ["gemma3", "gemma3_text"]:
            model_class = Gemma3ForCausalLM
            collate_fn  = make_collate_function_raw_text_dynamic_padding_gemma3(
                abstracts, tokenizer, max_seq_length=max_seq_length, on_overflow=tcfg.get("on_context_overflow", "warn"))
        else:
            raise ValueError(f"ERROR: Model type {config.model_type} not supported")

        # real length enforcement lives in the collate function above (TRL's own
        # max_seq_length is inert under dataset_kwargs={"skip_prepare_dataset": True});
        # this pre-pass just reports how bad the overflow would be before training starts
        emb_tok_id = TYPE_TOKEN_MAP_DICT[config.model_type]["emb_tok_id"]
        sample_prompt_ids = training_dataset[0]["prompt_ids"]
        prompt_overhead_tokens = len(sample_prompt_ids) - sample_prompt_ids.count(emb_tok_id)
        abstract_token_lens = get_or_compute_abstract_token_lengths(
            abstracts, tokenizer, cache_path=graph_outputd / f"abstract_token_lens_{config.model_type}.npy")
        context_length_stats = compute_context_length_stats(
            training_dataset, abstract_token_lens, prompt_overhead_tokens, max_seq_length,
            include_target=True, target_token_lens=abstract_token_lens)
        print(f"Raw-text context length stats (train split): {context_length_stats}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "context_length_stats.json", "w") as f:
            json.dump(context_length_stats, f, indent=2)
    elif config.model_type == "llama":
        model_class = LlamaForEmbeddingLM
        collate_fn  = make_collate_function_dynamic_padding_llama(embeddings, abstracts, tokenizer)
    elif config.model_type in ["gemma3", "gemma3_text"]:
        model_class = Gemma3ForEmbeddingLM
        collate_fn  = make_collate_function_dynamic_padding_gemma3(embeddings, abstracts, tokenizer)
    else:
        raise ValueError(f"ERROR: Model type {config.model_type} not supported")

    if not tcfg.get("finetune", True):
        print("finetune=false: skipping training (evaluate_model.py will load the base model directly)")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"[rank {local_rank}] pinned to cuda:{local_rank}")

    elm = model_class.from_pretrained(
        tcfg.basemodel_path,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank}
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj"],
        # "adapter" only exists on the embedding-injection model class -- a plain
        # LlamaForCausalLM/Gemma3ForCausalLM (raw_text mode) has no such module,
        # and get_peft_model() would crash trying to resolve it
        modules_to_save=["adapter"] if context_mode == "embedding" else None,
    )

    elm_lora = get_peft_model(elm, peft_config)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = tcfg.batch_size * tcfg.gradient_accumulation_steps * world_size
    num_training_steps   = (tcfg.num_train_epochs * len(training_dataset)) // effective_batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir + "/logs",
        per_device_train_batch_size=tcfg.batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        learning_rate=tcfg.learning_rate,
        max_grad_norm=1.0,
        save_steps=tcfg.save_steps,
        max_steps=num_training_steps,
        eval_steps=tcfg.eval_steps,
        logging_steps=tcfg.eval_steps,
        remove_unused_columns=False,
        bf16=True,
    )

    trainer = SFTTrainer(
        elm_lora,
        train_dataset=training_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=collate_fn,
        max_seq_length=max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        callbacks=[ResourceLoggingCallback()],
    )

    resume_checkpoint = None
    if tcfg.resume_from_checkpoint == "latest":
        resume_checkpoint = True
        print("Resuming from the latest checkpoint")
    elif tcfg.resume_from_checkpoint:
        resume_checkpoint = tcfg.resume_from_checkpoint
        print(f"Resuming from specified checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

if __name__ == "__main__":
    main()
