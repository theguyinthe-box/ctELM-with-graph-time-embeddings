from transformers import LlamaForCausalLM, LlamaConfig
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from typing import Optional
import torch
from openelm.tokens_map import TOKEN_MAP_DICT

class EmbeddingLMConfigMixin:    
    # we cannot use model_type = "llama_embedding" because it is not registered
    # the whole huggingface ecosystem (peft, trl, etc.) cannot recognize it as a valid model
    # model_type = "llama_embedding"
    
    def __init__(
        self,
        dim_embed_domain=1024,
        dim_adapter_hidden=2048,
        pretrained_model_name_or_path=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim_embed_domain = dim_embed_domain
        self.dim_adapter_hidden = dim_adapter_hidden
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

class LlamaForEmbeddingConfig(LlamaConfig, EmbeddingLMConfigMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "llama"

class Gemma3ForEmbeddingConfig(EmbeddingLMConfigMixin, Gemma3TextConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "gemma3"

class EmbeddingLMMixin:
    def __init__(self, config):
        # LlamaForCausalLM.__init__(self, config)
        super().__init__(config) 
        self.dim_embed_domain = config.dim_embed_domain
        self.dim_adapter_hidden = config.dim_adapter_hidden
        # Get the dimension of the token embeddings
        self.dim_embed_token = self.model.embed_tokens.embedding_dim

        # Initialize adapter in __init__
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(self.dim_embed_domain, self.dim_adapter_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dim_adapter_hidden, self.dim_embed_token)
        )
        
        # Load token mapping from TOKEN_MAP_DICT
        pretrained_model_name_or_path = getattr(self.config, 'pretrained_model_name_or_path', None)
        if pretrained_model_name_or_path and pretrained_model_name_or_path in TOKEN_MAP_DICT:
            token_map = TOKEN_MAP_DICT[pretrained_model_name_or_path]
            self.emb_tok_id = token_map['emb_tok_id']
            self.gen_tok_id = token_map['gen_tok_id']
            self.emb_tok = token_map['emb_tok']
            self.gen_tok = token_map['gen_tok']
        else:
            raise ValueError(
                f"Model name '{pretrained_model_name_or_path}' not found in TOKEN_MAP_DICT. "
                f"Available models: {list(TOKEN_MAP_DICT.keys())}"
            )
    
    def forward(
        self,
        *args,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        domain_embeddings=None,
        **kwargs
    ):
        # start with embedding input_ids 
        embs = self.model.embed_tokens(input_ids)
        emb_i = 0
        tot_emb = 0
        
        # loop each batch
        for i in range(embs.shape[0]):
            # loop each instance in a batch
            for j in range(embs.shape[1]):
                # replace the embed_token's embedding with domain embedding
                if input_ids[i,j] == self.emb_tok_id:
                    embs[i,j] = self.adapter(domain_embeddings[emb_i])
                    emb_i += 1

        kwargs['inputs_embeds']=embs
        kwargs['input_ids']=None
        # pass the modified embeddings to the parent class's forward function
        # this allows we 
        # -> pass the modified embeddings through transformer layers
        # -> apply language modeling head
        # -> generate output
        return super().forward(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        *args,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        domain_embeddings=None,
        **kwargs
    ):
        output = super().prepare_inputs_for_generation(*args, **kwargs)
        # ensure that domain embeddings are passed through each generation step
        output.update({"domain_embeddings": domain_embeddings})
        # this output will be used in forward function
        return output

class LlamaForEmbeddingLM(EmbeddingLMMixin, LlamaForCausalLM):
    #config_class = LlamaForEmbeddingConfig
    pass

class Gemma3ForEmbeddingLM(EmbeddingLMMixin, Gemma3ForCausalLM):
    #config_class = Gemma3ForEmbeddingConfig
    pass

# helper function to initialize the embedding model from a causal LM checkpoint
def initialize_embedding_model_from_causal_lm(
    pretrained_model_name_or_path,
    dim_embed_domain=1024,
    dim_adapter_hidden=2048
):
    """
    Load a LlamaForEmbeddingLM model with adapter weights from a LlamaForCausalLM checkpoint.
    
    Args:
        pretrained_model_name_or_path (str): Path to the base LlamaForCausalLM model
        dim_embed_domain (int): Domain embedding dimension
        dim_adapter_hidden (int): Adapter hidden dimension
        **kwargs: Additional args to pass to from_pretrained (like device_map, torch_dtype, etc.)
        
    Returns:
        LlamaForEmbeddingLM: Loaded model with adapter initialized
    """
    
    if "Llama" in pretrained_model_name_or_path:
        config_class = LlamaForEmbeddingConfig
        config_base_class = LlamaConfig
        model_class = LlamaForEmbeddingLM
        model_base_class = LlamaForCausalLM
    elif "gemma" in pretrained_model_name_or_path:
        config_class = Gemma3ForEmbeddingConfig
        config_base_class = Gemma3TextConfig
        model_class = Gemma3ForEmbeddingLM
        model_base_class = Gemma3ForCausalLM
    else:
        print(f"ERROR: could not infer model type from '{pretrained_model_name_or_path}'")
        return None

    # Load the original config from the pretrained model
    original_config = config_base_class.from_pretrained(pretrained_model_name_or_path)

    # Create embedding config based on the original config
    embedding_config = config_class(
        **original_config.to_dict(),
        dim_embed_domain=dim_embed_domain,
        dim_adapter_hidden=dim_adapter_hidden,
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    
    print("Initializing embedding model with the new config (adapter will be initialized randomly)")
    # Initialize our embedding model with the new config (adapter will be initialized randomly)
    embedding_model = model_class(embedding_config)
    
    print("Loading the weights from the pretrained model")
    # Load the weights from the pretrained model
    causal_lm_model = model_base_class.from_pretrained(
        pretrained_model_name_or_path
    )
    
    print("Getting the state dict from the pretrained model")
    # Get the state dict from the causal LM model
    causal_lm_state_dict = causal_lm_model.state_dict()
    
    # Filter out any keys that might not match our model (should be none in this case)
    embedding_model_keys = set(embedding_model.state_dict().keys())
    filtered_state_dict = {
        k: v for k, v in causal_lm_state_dict.items() 
        if k in embedding_model_keys and not k.startswith('adapter.')
    }

    print("Loading the state dict into our embedding model")
    # Load the filtered state dict into our embedding model
    # strict=False: ignore the missing keys, e.g., adapter weights
    embedding_model.load_state_dict(filtered_state_dict, strict=False)
    
    return embedding_model