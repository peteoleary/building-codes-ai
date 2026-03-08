"""
Fine-tune GPT-2 model on San Francisco Building Code data
"""
import os
import torch
import tiktoken
from torch.utils.data import DataLoader
from gpt_model import GPTModel
from gpt_dataset import GPTDatasetV1
from gpt_utils import generate, text_to_token_ids, token_ids_to_text
import time

# Model configurations for GPT-2 Small and Medium
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension (Small: 768)
    "n_heads": 12,          # Number of attention heads (Small: 12)
    "n_layers": 12,         # Number of layers (Small: 12)
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True        # Query-Key-Value bias (GPT-2 uses True)
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 1024,        # Embedding dimension (Medium: 1024)
    "n_heads": 16,          # Number of attention heads (Medium: 16)
    "n_layers": 24,         # Number of layers (Medium: 24)
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True        # Query-Key-Value bias (GPT-2 uses True)
}

# Model configurations to train
MODEL_CONFIGS = {
    "small": {
        "config": GPT_CONFIG_124M,
        "pretrained_path": "gpt2-small-124M.pth",
        "output_path": "gpt2-san-francisco-finetuned.pth",
        "checkpoint_path": "gpt2-small-san-francisco-checkpoint.pth",
        "batch_size": 4,
        "name": "GPT-2 Small (124M)"
    },
    "medium": {
        "config": GPT_CONFIG_355M,
        "pretrained_path": "gpt2-medium-355M.pth",
        "output_path": "gpt2-medium-san-francisco-finetuned.pth",
        "checkpoint_path": "gpt2-medium-san-francisco-checkpoint.pth",
        "batch_size": 2,
        "name": "GPT-2 Medium (355M)"
    }
}

# Training hyperparameters
MAX_LENGTH = 256  # Sequence length for training
STRIDE = 128      # Stride for sliding window
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5  # Lower learning rate for fine-tuning
WEIGHT_DECAY = 0.1
EVAL_FREQ = 50    # Evaluate every N steps
EVAL_ITER = 5     # Number of batches to use for evaluation


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over the data loader."""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on training and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print a sample text."""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, 
            idx=encoded,
            max_new_tokens=100, 
            context_size=context_size,
            temperature=0.7,
            top_k=50
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("\nGenerated sample:")
    print("-" * 80)
    print(decoded_text)
    print("-" * 80)
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    """Main training loop."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1}/{num_epochs} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
        # Generate sample after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def setup_layer_freezing(model, strategy="freeze_early"):
    """
    Setup layer freezing with different strategies.
    
    Args:
        model: GPTModel instance
        strategy: One of:
            - "freeze_early": Freeze first 75% of layers (recommended)
            - "freeze_most": Only train last 2 layers + head
            - "head_only": Only train output head
            - "no_freeze": Train everything (default behavior)
    """
    if strategy == "no_freeze":
        print("✓ Training all layers (no freezing)")
        return
    
    num_layers = len(model.trf_blocks)
    
    if strategy == "head_only":
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze only output head
        for param in model.out_head.parameters():
            param.requires_grad = True
        print("✓ Only training output head")
    
    elif strategy == "freeze_most":
        # Freeze embeddings
        for param in model.tok_emb.parameters():
            param.requires_grad = False
        for param in model.pos_emb.parameters():
            param.requires_grad = False
        # Freeze all but last 2 transformer blocks
        for i in range(num_layers - 2):
            for param in model.trf_blocks[i].parameters():
                param.requires_grad = False
        print(f"✓ Training last 2 transformer blocks + output layers")
    
    elif strategy == "freeze_early":
        # Freeze 75% of layers (train last 25%)
        freeze_count = int(num_layers * 0.75)
        
        # Freeze embeddings
        for param in model.tok_emb.parameters():
            param.requires_grad = False
        for param in model.pos_emb.parameters():
            param.requires_grad = False
        
        # Freeze first 75% of transformer blocks
        for i in range(freeze_count):
            for param in model.trf_blocks[i].parameters():
                param.requires_grad = False
        
        print(f"✓ Frozen first {freeze_count}/{num_layers} transformer blocks")
        print(f"✓ Training last {num_layers - freeze_count} blocks + output layers")


def train_single_model(model_key, model_info, train_loader, val_loader, device, tokenizer):
    """Train a single model configuration."""
    print("\n" + "="*80)
    print(f"TRAINING {model_info['name']}")
    print("="*80)
    
    config = model_info["config"]
    pretrained_path = model_info["pretrained_path"]
    output_path = model_info["output_path"]
    checkpoint_path = model_info["checkpoint_path"]
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model from {pretrained_path}...")
    model = GPTModel(config)
    
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, weights_only=True))
        print("✓ Pre-trained model loaded successfully")
    else:
        print(f"Warning: {pretrained_path} not found. Training from scratch.")
    
    # Setup layer freezing
    print("\nSetting up layer freezing...")
    setup_layer_freezing(model, strategy="freeze_early")
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Calculate initial loss
    print("\nCalculating initial loss...")
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=EVAL_ITER)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=EVAL_ITER)
    print(f"Initial training loss: {train_loss:.3f}")
    print(f"Initial validation loss: {val_loss:.3f}")
    
    # Train the model
    start_time = time.time()
    start_context = "The San Francisco Building Code"
    
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=NUM_EPOCHS, 
        eval_freq=EVAL_FREQ, 
        eval_iter=EVAL_ITER,
        start_context=start_context, 
        tokenizer=tokenizer
    )
    
    training_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"{'='*80}")
    
    # Save the fine-tuned model
    print(f"\nSaving fine-tuned model to {output_path}...")
    torch.save(model.state_dict(), output_path)
    print("✓ Model saved successfully")
    
    # Save checkpoint
    print(f"Saving checkpoint to {checkpoint_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tokens_seen': tokens_seen,
        'config': config
    }, checkpoint_path)
    print("✓ Checkpoint saved successfully")
    
    # Final generation test
    print("\n" + "="*80)
    print("FINAL MODEL TEST - Generating sample:")
    print("="*80)
    model.eval()
    test_prompts = [
        "The San Francisco Building Code",
        "Section 101",
        "Building construction"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        encoded = text_to_token_ids(prompt, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=encoded,
                max_new_tokens=100,
                context_size=config["context_length"],
                temperature=0.7,
                top_k=50
            )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text)
    
    print(f"\n✓ {model_info['name']} training complete!")
    return training_time


def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print("="*80)
    print("GPT-2 FINE-TUNING ON SAN FRANCISCO BUILDING CODES")
    print("="*80)
    print(f"Using device: {device}")
    print(f"Models to train: {', '.join([MODEL_CONFIGS[k]['name'] for k in ['small', 'medium']])}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load training data
    print("\nLoading training data from san_francisco-ca-1.txt...")
    with open("san_francisco-ca-1.txt", "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    print(f"Total characters: {len(text):,}")
    print(f"Total words (approx): {len(text.split()):,}")
    
    # Tokenize and check length
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(f"Total tokens: {len(token_ids):,}")
    
    # Split into train and validation (90/10 split)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    print(f"\nTraining set characters: {len(train_text):,}")
    print(f"Validation set characters: {len(val_text):,}")
    
    # Create base datasets
    print("\nCreating datasets...")
    train_dataset = GPTDatasetV1(train_text, tokenizer, MAX_LENGTH, STRIDE)
    val_dataset = GPTDatasetV1(val_text, tokenizer, MAX_LENGTH, STRIDE)
    
    # Train both models
    total_start_time = time.time()
    training_times = {}
    
    for model_key in ["small", "medium"]:
        model_info = MODEL_CONFIGS[model_key]
        batch_size = model_info["batch_size"]
        
        # Create dataloaders with appropriate batch size for this model
        print(f"\nCreating dataloaders for {model_info['name']} (batch_size={batch_size})...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Train this model
        model_time = train_single_model(
            model_key, 
            model_info, 
            train_loader, 
            val_loader, 
            device, 
            tokenizer
        )
        training_times[model_key] = model_time
    
    # Final summary
    total_time = time.time() - total_start_time
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    print(f"\nIndividual model times:")
    for model_key, model_time in training_times.items():
        print(f"  {MODEL_CONFIGS[model_key]['name']}: {model_time/60:.2f} minutes")
    print("\nFine-tuned models ready for use:")
    for model_key in ["small", "medium"]:
        print(f"  ✓ {MODEL_CONFIGS[model_key]['output_path']}")
    print("="*80)


if __name__ == "__main__":
    main()
