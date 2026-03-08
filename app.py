# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit as cl
from chainlit.input_widget import Select


# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from gpt_model import GPTModel
from gpt_utils import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

# Set device with MPS support for Apple Silicon
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# GPT-2 Model Configurations
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,         # Embedding dimension (Medium)
    "n_heads": 16,           # Number of attention heads (Medium)
    "n_layers": 24,          # Number of layers (Medium)
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}


def load_model(model_path, config):
    """
    Load a GPT-2 model from a checkpoint file with the specified config.
    
    Args:
        model_path: Path to the model checkpoint
        config: Model configuration dict (GPT_CONFIG_124M or GPT_CONFIG_355M)
    
    Returns:
        Tuple of (model, config) or (None, None) if loading fails
    """
    if not Path(model_path).exists():
        print(f"Warning: Could not find {model_path}")
        return None, None
    
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        model = GPTModel(config)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"✓ Loaded {model_path} successfully on {device}")
        return model, config
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None, None


# Load all available models at startup
print("\nLoading models...")
model_configs = {
    "GPT-2 Small (Pretrained)": ("gpt2-small-124M.pth", GPT_CONFIG_124M),
    "GPT-2 Small (San Francisco Fine-tuned)": ("gpt2-san-francisco-finetuned.pth", GPT_CONFIG_124M),
    "GPT-2 Medium (Pretrained)": ("gpt2-medium-355M.pth", GPT_CONFIG_355M),
    "GPT-2 Medium (San Francisco Fine-tuned)": ("gpt2-medium-san-francisco-finetuned.pth", GPT_CONFIG_355M),
}

models = {}
model_configs_loaded = {}

for name, (path, config) in model_configs.items():
    model, loaded_config = load_model(path, config)
    if model is not None:
        models[name] = model
        model_configs_loaded[name] = loaded_config

if not models:
    print("Error: No models could be loaded!")
    sys.exit(1)

print(f"\n{len(models)} model(s) loaded successfully")
print(f"Available models: {list(models.keys())}\n")

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")


@cl.on_chat_start
async def start():
    """
    Initialize chat session with settings.
    """
    # Get the first available model as default
    default_model = list(models.keys())[0]
    
    # Store the selected model in user session
    cl.user_session.set("selected_model", default_model)
    
    # Create settings with model dropdown
    settings = await cl.ChatSettings(
        [
            Select(
                id="model_selection",
                label="Select Model",
                values=list(models.keys()),
                initial_value=default_model,
            )
        ]
    ).send()
    
    await cl.Message(
        content=f"Welcome! Currently using: **{default_model}**\n\nYou can switch models using the settings icon."
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    """
    Handle model selection changes.
    """
    selected_model = settings["model_selection"]
    cl.user_session.set("selected_model", selected_model)
    
    await cl.Message(
        content=f"Switched to: **{selected_model}**"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    The main Chainlit function.
    """
    try:
        # Get the currently selected model
        selected_model_name = cl.user_session.get("selected_model")
        if not selected_model_name:
            selected_model_name = list(models.keys())[0]
        
        model = models[selected_model_name]
        config = model_configs_loaded[selected_model_name]
        
        # Use user's message directly as prompt
        prompt = message.content
        
        print(f"Model: {selected_model_name}")
        print(f"Received prompt: {prompt}")
        
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=150,  # Increased for better responses
            context_size=config["context_length"],
            temperature=0.7,  # Add some randomness
            top_k=50,  # Use top-k sampling
            eos_id=50256
        )

        response = token_ids_to_text(token_ids, tokenizer)
        print(f"Generated response: {response}\n")
        
        await cl.Message(
            content=response,
        ).send()
    
    except Exception as e:
        print(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        await cl.Message(
            content=f"Error: {str(e)}",
        ).send()