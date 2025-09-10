from safetensors.torch import load_file
from peft import PeftModel
import argparse
import yaml
import os
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

def load_custom_embeddings_qwen(model, tokenizer, embed_path, lm_head_path):
    """Load custom embedding layers"""
    # Load modified embed_token
    embed_tokens_state_dict = load_file(embed_path)
    print(embed_tokens_state_dict.keys())
    # embed_tokens_state_dict = {"weight": embed_tokens_state_dict["model.embed_tokens.weight"]}
    embed_tokens = torch.nn.Embedding(151936, 2560, tokenizer.pad_token_id, dtype=torch.bfloat16)
    print(embed_tokens_state_dict["weight"].shape)
    print(embed_tokens_state_dict["weight"].dtype)

    embed_tokens.load_state_dict(embed_tokens_state_dict)
    model.embed_tokens = embed_tokens.bfloat16().to('cuda')

    torch.cuda.empty_cache()
    
    return model

def load_local_tokenizer(tokenizer_path, max_length=4096):
    """Load and configure tokenizer"""
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, trust_remote_code=True)

    # Add all special tokens consistently with tokenize_test.py
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token" : "<|endoftext|>",
        "pad_token" : "<|endoftext|>",
    }
    tokenizer.add_special_tokens(special_tokens)

    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")
    print(f"Tokenizer eos token ID: {tokenizer.eos_token_id}")
    tokenizer.model_max_length = max_length
    return tokenizer


def load_and_merge_model(base_model_id, model_path, tokenizer_path, max_length):
    """Load and merge PEFT model"""
    print(f"Loading PEFT model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    tokenizer = load_local_tokenizer(tokenizer_path, max_length)
    
    model.gradient_checkpointing_enable({"use_reentrant":False})
    
    embed_path = f"{model_path}/embed_tokens.safetensors"
    
    model = load_custom_embeddings_qwen(model, tokenizer, embed_path, lm_head_path=embed_path)
    
    model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.merge_and_unload()
    
    return model, tokenizer

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    
    argparser = argparse.ArgumentParser(description="Evaluate a language model using lm-eval-harness")
    argparser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = argparser.parse_args()
    
    config = load_config(args.config)
    
    model, tokenizer = load_and_merge_model(
        config["base_model_id"],
        config["model_path"],
        config["tokenizer_path"],
        config.get("max_length", 4096),
    )
    
    if not os.path.exists(config["output_model_path"]):
        os.makedirs(config["output_model_path"])
        
    model.save_pretrained(config["output_model_path"])
    tokenizer.save_pretrained(config["output_model_path"])
    
if __name__ == "__main__":
    main()