import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
from openelm.tokens_map import TYPE_TOKEN_MAP_DICT
from openelm.model import LlamaForEmbeddingLM, Gemma3ForEmbeddingLM

##########
## Collate function for dynamic padding
##########
def make_collate_function_dynamic_padding_gemma3(embeddings, abstracts, tokenizer):
    return make_collate_function_dynamic_padding(embeddings, abstracts, tokenizer, model="gemma3")

def make_collate_function_dynamic_padding_llama(embeddings, abstracts, tokenizer):
    return make_collate_function_dynamic_padding(embeddings, abstracts, tokenizer, model="llama")

def make_collate_function_dynamic_padding(embeddings, abstracts, tokenizer, model="llama"):
    """
    Build a collate function that lazily resolves the target abstract's tokens
    (and domain embedding vectors) at batch time from a stored index, instead of
    requiring every dataset row to carry its own copy of them. The same target
    abstract is the endpoint of many different citation chains, so baking its
    tokenized text into every row duplicates it across the whole dataset;
    resolving — and caching — it here keeps that cost to one tokenization per
    distinct target regardless of how many rows reference it.
    """
    pad_token_id = TYPE_TOKEN_MAP_DICT[model]["pad_tok_id"]
    gen_token_id = TYPE_TOKEN_MAP_DICT[model]["gen_tok_id"]
    gen_token    = TYPE_TOKEN_MAP_DICT[model]["gen_tok"]

    # the tokens a chat template appends after the assistant's content (e.g.
    # end-of-turn + EOS) are fixed regardless of what that content is, so derive
    # them once here rather than persisting them redundantly in every row
    probe_ids = tokenizer.apply_chat_template([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": gen_token},
    ])
    closing_ids = probe_ids[probe_ids.index(gen_token_id) + 1:]

    target_ids_cache = {}
    def resolve_target_ids(target_idx):
        if target_idx not in target_ids_cache:
            target_ids_cache[target_idx] = tokenizer.encode(str(abstracts[target_idx]), add_special_tokens=False)
        return target_ids_cache[target_idx]

    def collate_fn(examples):
        input_ids = []
        labels = []
        sequences = []
        for example in examples:
            prompt_ids = example["prompt_ids"]
            target_ids = resolve_target_ids(example["target_idx"])
            # gen token itself is a boundary marker, not fed as an input token
            gen_tok_pos = len(prompt_ids) - 1
            ids_without_gen_token = prompt_ids[:-1] + target_ids + closing_ids
            sequences.append((gen_tok_pos, ids_without_gen_token))

        max_length = max(len(ids) for _, ids in sequences)

        for gen_tok_pos, ids_without_gen_token in sequences:
            # create an array with max_length, filled up by pad_token_id
            input_ids_padded = torch.full((max_length,), pad_token_id, dtype=torch.long)
            input_ids_padded[:len(ids_without_gen_token)] = torch.tensor(ids_without_gen_token)
            input_ids.append(input_ids_padded)

            labels_padded = torch.full((max_length,), -100, dtype=torch.long)
            # set prompt [:gen_tok_pos] as -100
            # set pads [len(ids_without_gen_token):] as -100
            # only learn target, which is [gen_tok_pos:len(ids_without_gen_token)]
            labels_padded[gen_tok_pos:len(ids_without_gen_token)] = input_ids_padded[gen_tok_pos:len(ids_without_gen_token)]
            labels.append(labels_padded)

        embs = [
            torch.tensor(embeddings[idx])
            for example in examples
            for idx in example["domain_embedding_idx"]
        ]

        return {"input_ids": torch.stack(input_ids), "domain_embeddings": embs, "labels": torch.stack(labels)}

    return collate_fn

##########
## Collate function for raw-text baseline (feeds context abstracts as
## literal tokens instead of adapter-compressed embeddings, so a plain
## causal LM can be trained/evaluated as a comparison point for ctELM)
##########
def expand_context_ids(prompt_ids, context_idx, abstracts, tokenizer, emb_tok_id, cache):
    """
    Walks prompt_ids left-to-right, replacing each emb_tok_id occurrence, in
    order, with the tokenized text of the corresponding context abstract
    (same left-to-right slot-order guarantee the embedding path's forward()
    relies on when it scans for emb_tok_id). `cache` is a caller-owned dict
    (node idx -> token id list) so a context abstract that recurs across
    many overlapping chains is tokenized once, same rationale as
    target_ids_cache above.

    Returns (expanded_ids, context_token_count) -- the second element is the
    exact "tokens spent on context" measurement, a free byproduct of
    expansion.
    """
    expanded = []
    context_token_count = 0
    ctx_i = 0
    for tok in prompt_ids:
        if tok == emb_tok_id:
            idx = context_idx[ctx_i]
            if idx not in cache:
                cache[idx] = tokenizer.encode(str(abstracts[idx]), add_special_tokens=False)
            ids = cache[idx]
            expanded.extend(ids)
            context_token_count += len(ids)
            ctx_i += 1
        else:
            expanded.append(tok)
    return expanded, context_token_count

def make_collate_function_raw_text_dynamic_padding_gemma3(abstracts, tokenizer, **kwargs):
    return make_collate_function_raw_text_dynamic_padding(abstracts, tokenizer, model="gemma3", **kwargs)

def make_collate_function_raw_text_dynamic_padding_llama(abstracts, tokenizer, **kwargs):
    return make_collate_function_raw_text_dynamic_padding(abstracts, tokenizer, model="llama", **kwargs)

def make_collate_function_raw_text_dynamic_padding(abstracts, tokenizer, model="llama", max_seq_length=None, on_overflow="warn"):
    """
    Raw-text analog of make_collate_function_dynamic_padding: splices each
    context abstract's own tokens in place of its emb_tok slot instead of
    injecting an adapter-compressed embedding, so a plain
    LlamaForCausalLM/Gemma3ForCausalLM (no domain_embeddings kwarg) can be
    trained on literal text. Returns {"input_ids", "labels"} only.

    max_seq_length / on_overflow provide real length enforcement -- TRL's
    own max_seq_length argument is inert here (SFTTrainer's
    _prepare_dataset returns immediately under
    dataset_kwargs={"skip_prepare_dataset": True} before max_seq_length is
    ever consumed), and raw-text sequences run far longer than the
    embedding path's, so something has to enforce it.
      - on_overflow="warn" (default): overlong sequences pass through
        unchanged; collate_fn.overflow_count / .total_count accumulate on
        the closure for the caller to log periodically.
      - on_overflow="drop": examples exceeding max_seq_length are excluded
        from the batch (never drops every example -- keeps at least one).
    """
    pad_token_id = TYPE_TOKEN_MAP_DICT[model]["pad_tok_id"]
    gen_token_id = TYPE_TOKEN_MAP_DICT[model]["gen_tok_id"]
    gen_token    = TYPE_TOKEN_MAP_DICT[model]["gen_tok"]
    emb_tok_id   = TYPE_TOKEN_MAP_DICT[model]["emb_tok_id"]

    probe_ids = tokenizer.apply_chat_template([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": gen_token},
    ])
    closing_ids = probe_ids[probe_ids.index(gen_token_id) + 1:]

    target_ids_cache  = {}
    context_ids_cache = {}
    def resolve_target_ids(target_idx):
        if target_idx not in target_ids_cache:
            target_ids_cache[target_idx] = tokenizer.encode(str(abstracts[target_idx]), add_special_tokens=False)
        return target_ids_cache[target_idx]

    def collate_fn(examples):
        sequences = []
        for example in examples:
            expanded_prompt, _ = expand_context_ids(
                example["prompt_ids"][:-1], example["domain_embedding_idx"],
                abstracts, tokenizer, emb_tok_id, context_ids_cache
            )
            target_ids = resolve_target_ids(example["target_idx"])
            gen_tok_pos = len(expanded_prompt)
            ids_without_gen_token = expanded_prompt + target_ids + closing_ids
            sequences.append((gen_tok_pos, ids_without_gen_token))

        if max_seq_length is not None:
            collate_fn.total_count += len(sequences)
            overflowing = sum(1 for _, ids in sequences if len(ids) > max_seq_length)
            collate_fn.overflow_count += overflowing
            if overflowing and on_overflow == "drop":
                keep = [i for i, (_, ids) in enumerate(sequences) if len(ids) <= max_seq_length]
                sequences = [sequences[i] for i in keep] or sequences[:1]
            elif overflowing and on_overflow not in ("warn", "drop"):
                raise ValueError(f"Unknown on_overflow policy: {on_overflow}")

        max_length = max(len(ids) for _, ids in sequences)

        input_ids = []
        labels = []
        for gen_tok_pos, ids_without_gen_token in sequences:
            input_ids_padded = torch.full((max_length,), pad_token_id, dtype=torch.long)
            input_ids_padded[:len(ids_without_gen_token)] = torch.tensor(ids_without_gen_token)
            input_ids.append(input_ids_padded)

            labels_padded = torch.full((max_length,), -100, dtype=torch.long)
            labels_padded[gen_tok_pos:len(ids_without_gen_token)] = input_ids_padded[gen_tok_pos:len(ids_without_gen_token)]
            labels.append(labels_padded)

        return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels)}

    collate_fn.overflow_count = 0
    collate_fn.total_count = 0
    return collate_fn

##########
## Helper function to load elm model
##########
def load_elm_model(configs):
    """
    Load the elm model from the config file.

    Args:
        configs: dictionary containing the config file

    Returns:
        tokenizer: tokenizer for the basemodel
        model_config: config for the basemodel
        lora_elm: basemodel with PEFT applied (elm = basemodel + PEFT)
    """

    # check if configs is a dictionary
    # if check if configs has backbone_model_path, peft_model_id, and device
    if not configs.get('backbone_model_path'):
        raise ValueError("backbone_model_path is required in the config file")
    if not configs.get('peft_model_id'):
        raise ValueError("peft_model_id is required in the config file")
    if not configs.get('device'):
        raise ValueError("device is required in the config file")

    print(f"Loading backbone model from {configs['backbone_model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(configs['backbone_model_path'])

    model_config = AutoConfig.from_pretrained(configs['backbone_model_path'])
    print(f"Backbone model type: {model_config.model_type}")

    # load elm model based on the base model type
    if model_config.model_type == "llama":
        model_class = LlamaForEmbeddingLM
    elif model_config.model_type in ["gemma3", "gemma3_text"]:
        model_class = Gemma3ForEmbeddingLM
    elm = model_class.from_pretrained(
        configs['backbone_model_path'], 
        torch_dtype=torch.bfloat16,
        device_map=configs['device'])

    print(f"Loading PEFT model from {configs['peft_model_id']}")
    lora_elm = PeftModel.from_pretrained(elm, configs['peft_model_id'])
    lora_elm = lora_elm.merge_and_unload()

    # ensure eos_token_id is set correctly based on the base model type
    if model_config.model_type == "llama":
        lora_elm.config.eos_token_id = tokenizer.eos_token_id
    elif model_config.model_type in ["gemma3", "gemma3_text"]:
        lora_elm.config.eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    print(f"EOS token ID: {lora_elm.config.eos_token_id}")
    
    return tokenizer, model_config, lora_elm

##########
## Helper function to generate input for inference
##########
def batched_inference_input_generator(prompt_template, emb_token, tokenizer, embeddings, batch_size=8, device="cuda"):
    """
    Generate input for inference in batches.

    Args:
        prompt_template: str
        emb_token: str
        tokenizer: AutoTokenizer
        embeddings: dictionary of numpy arrays
        batch_size: int
        device: str
    """
    # Check the number of embeddings and number of "emb_token" in the prompt template are matched
    if prompt_template.count("{emb_token}") != len(embeddings):
        raise ValueError("The number of embeddings and the number of 'emb_token' in the prompt template are not matched")
    
    # Check the number of rows in each embedding file are matched
    number_of_rows = len(embeddings[0])
    for j in range(len(embeddings)):
        if len(embeddings[j]) != number_of_rows:
            raise ValueError("The number of rows in the embedding files are not matched")
    
    # Process in batches
    for batch_start in range(0, number_of_rows, batch_size):
        batch_end = min(batch_start + batch_size, number_of_rows)
        batch_indices = range(batch_start, batch_end)
        
        # Generate input ids for this batch
        input_ids_list = []
        for i in batch_indices:        
            input_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt_template.format(emb_token=emb_token)},
                ],
                return_tensors="pt", add_generation_prompt=True
            ).to(device)
            input_ids_list.append(input_ids)
        
        # Pad input ids for this batch
        max_length = max(ids.shape[1] for ids in input_ids_list)
        padded_inputs = []
        prompt_lengths = []
        for input_ids in input_ids_list:
            # Pad to max length within this batch
            prompt_length = input_ids.shape[1]
            prompt_lengths.append(prompt_length)
            padding_length = max_length - prompt_length
            if padding_length > 0:
                padding = torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
                padded_input = torch.cat([input_ids, padding], dim=1)
            else:
                padded_input = input_ids
            
            padded_inputs.append(padded_input)
            
        # Stack all inputs into a batch
        batch_input_ids = torch.cat(padded_inputs, dim=0)
        # Convert embeddings to tensor (interleaved by row)
        batch_embs_tensor = []
        for i in batch_indices:
            for _, emb in embeddings.items():
                emb_tensor = torch.tensor(emb[i], dtype=torch.bfloat16).to(device)
                batch_embs_tensor.append(emb_tensor)
        
        yield batch_input_ids, batch_embs_tensor, prompt_lengths