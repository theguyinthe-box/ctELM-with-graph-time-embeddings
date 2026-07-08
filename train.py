import os
import argparse
from pathlib import Path
import numpy as np
from openelm.model import LlamaForEmbeddingLM, Gemma3ForEmbeddingLM
from openelm.utils import make_collate_function_dynamic_padding_llama, make_collate_function_dynamic_padding_gemma3
from openelm.config import load_config
from datasets import Dataset
from transformers import TrainingArguments, AutoConfig, AutoTokenizer
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

    config = AutoConfig.from_pretrained(tcfg.basemodel_path)
    if config.model_type == "llama":
        model_class = LlamaForEmbeddingLM
        collate_fn  = make_collate_function_dynamic_padding_llama(embeddings, abstracts, tokenizer)
    elif config.model_type in ["gemma3", "gemma3_text"]:
        model_class = Gemma3ForEmbeddingLM
        collate_fn  = make_collate_function_dynamic_padding_gemma3(embeddings, abstracts, tokenizer)
    else:
        raise ValueError(f"ERROR: Model type {config.model_type} not supported")

    elm = model_class.from_pretrained(
        tcfg.basemodel_path,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()}
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj"],
        modules_to_save=["adapter"],
    )

    elm_lora = get_peft_model(elm, peft_config)
    print(elm_lora.print_trainable_parameters())

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"We will train the model using {world_size} process(es).")
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
        max_seq_length=2048,
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
