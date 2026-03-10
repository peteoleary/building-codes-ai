"""
Train Sparse Autoencoders on GPT-2 Models

This script trains sparse autoencoders to discover interpretable features
in your pretrained and fine-tuned GPT-2 models. Based on OpenAI's approach.

Usage:
    bin/python train_sparse_autoencoder.py --model small --checkpoint pretrained
    bin/python train_sparse_autoencoder.py --model small --checkpoint finetuned
    bin/python train_sparse_autoencoder.py --model medium --checkpoint pretrained
    bin/python train_sparse_autoencoder.py --model medium --checkpoint finetuned
"""

import torch
import tiktoken
import argparse
from pathlib import Path
from gpt_model import GPTModel
import numpy as np
from tqdm import tqdm

# Model configurations
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,  # No dropout for inference
    "qkv_bias": True
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,  # No dropout for inference
    "qkv_bias": True
}


class SparseAutoencoder(torch.nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features.
    
    Architecture:
        - Encoder: Linear layer with ReLU activation
        - Decoder: Linear layer
        - L1 penalty on activations for sparsity
    """
    
    def __init__(self, input_dim, hidden_dim, l1_coeff=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff
        
        # Encoder: input_dim -> hidden_dim
        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: hidden_dim -> input_dim
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize decoder as transpose of encoder
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        
    def encode(self, x):
        """Encode input to sparse features"""
        return torch.relu(self.encoder(x))
    
    def decode(self, z):
        """Decode features back to input space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def loss(self, x, x_reconstructed, z):
        """
        Compute total loss = reconstruction loss + L1 sparsity penalty
        """
        # MSE reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        
        # L1 sparsity penalty on activations
        sparsity_loss = self.l1_coeff * torch.abs(z).sum(dim=1).mean()
        
        return reconstruction_loss + sparsity_loss, reconstruction_loss, sparsity_loss


def load_model(model_size, checkpoint_type, device):
    """Load a GPT-2 model"""
    if model_size == "small":
        config = GPT_CONFIG_124M
        if checkpoint_type == "pretrained":
            checkpoint_path = "gpt2-small-124M.pth"
        else:
            checkpoint_path = "gpt2-san-francisco-finetuned.pth"
    else:  # medium
        config = GPT_CONFIG_355M
        if checkpoint_type == "pretrained":
            checkpoint_path = "gpt2-medium-355M.pth"
        else:
            checkpoint_path = "gpt2-medium-san-francisco-finetuned.pth"
    
    print(f"Loading {model_size} {checkpoint_type} model from {checkpoint_path}...")
    model = GPTModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    return model, config


def collect_activations(model, config, num_samples=10000, layer_idx=6, device='cpu'):
    """
    Collect activations from a specific layer of the model.
    
    We'll collect from the residual stream after the MLP at layer_idx.
    """
    print(f"\nCollecting activations from layer {layer_idx}...")
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load text data
    data_file = "san_francisco-ca-1.txt"
    with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}")
    
    activations = []
    batch_size = 8
    max_length = 256
    
    # Hook to capture activations
    captured_acts = []
    
    def hook_fn(module, input, output):
        # output shape: (batch_size, seq_len, emb_dim)
        captured_acts.append(output.detach())
    
    # Register hook on the transformer block
    hook = model.trf_blocks[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for i in tqdm(range(0, min(num_samples * max_length, len(tokens) - max_length), max_length * batch_size)):
            batch_indices = []
            for j in range(batch_size):
                start_idx = i + j * max_length
                if start_idx + max_length < len(tokens):
                    batch_indices.append(tokens[start_idx:start_idx + max_length])
            
            if not batch_indices:
                break
            
            # Convert to tensor
            input_ids = torch.tensor(batch_indices, dtype=torch.long).to(device)
            
            # Forward pass
            captured_acts.clear()
            _ = model(input_ids)
            
            # Collect activations
            if captured_acts:
                acts = captured_acts[0]  # (batch_size, seq_len, emb_dim)
                # Reshape to (batch_size * seq_len, emb_dim)
                acts = acts.reshape(-1, acts.shape[-1])
                activations.append(acts.cpu())
            
            if len(activations) * batch_size * max_length >= num_samples * max_length:
                break
    
    hook.remove()
    
    # Concatenate all activations
    activations = torch.cat(activations, dim=0)
    print(f"✓ Collected {activations.shape[0]} activation vectors of dimension {activations.shape[1]}")
    
    return activations


def train_autoencoder(activations, hidden_dim=8192, l1_coeff=1e-3, 
                     num_epochs=10, batch_size=256, lr=1e-3, device='cpu'):
    """Train sparse autoencoder on activations"""
    
    input_dim = activations.shape[1]
    print(f"\nTraining Sparse Autoencoder:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  L1 coefficient: {l1_coeff}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Create autoencoder
    autoencoder = SparseAutoencoder(input_dim, hidden_dim, l1_coeff).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    # Training loop
    num_batches = len(activations) // batch_size
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_sparsity = 0
        
        # Shuffle activations
        perm = torch.randperm(len(activations))
        activations_shuffled = activations[perm]
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch = activations_shuffled[start_idx:end_idx].to(device)
            
            # Forward pass
            x_reconstructed, z = autoencoder(batch)
            
            # Compute loss
            loss, recon_loss, sparsity_loss = autoencoder.loss(batch, x_reconstructed, z)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_sparsity += (z > 0).float().mean().item()  # Fraction of active features
        
        # Print epoch stats
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_sparsity = epoch_sparsity / num_batches
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Sparsity={avg_sparsity:.3f}")
    
    return autoencoder


def main():
    parser = argparse.ArgumentParser(description="Train sparse autoencoder on GPT-2 activations")
    parser.add_argument("--model", choices=["small", "medium"], required=True,
                       help="Model size")
    parser.add_argument("--checkpoint", choices=["pretrained", "finetuned"], required=True,
                       help="Checkpoint type")
    parser.add_argument("--layer", type=int, default=6,
                       help="Layer to extract activations from (default: 6)")
    parser.add_argument("--hidden_dim", type=int, default=8192,
                       help="Sparse autoencoder hidden dimension (default: 8192)")
    parser.add_argument("--l1_coeff", type=float, default=1e-3,
                       help="L1 sparsity coefficient (default: 1e-3)")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of activation samples to collect (default: 10000)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs (default: 10)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.model, args.checkpoint, device)
    
    # Collect activations
    activations = collect_activations(
        model, config, 
        num_samples=args.num_samples,
        layer_idx=args.layer,
        device=device
    )
    
    # Train autoencoder
    autoencoder = train_autoencoder(
        activations,
        hidden_dim=args.hidden_dim,
        l1_coeff=args.l1_coeff,
        num_epochs=args.epochs,
        device=device
    )
    
    # Save autoencoder with metadata
    output_name = f"sae_{args.model}_{args.checkpoint}_layer{args.layer}.pth"
    checkpoint = {
        'model_state_dict': autoencoder.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'layer': args.layer,
        'model_size': args.model,
        'checkpoint_type': args.checkpoint,
        'l1_coeff': args.l1_coeff,
    }
    torch.save(checkpoint, output_name)
    print(f"\n✓ Sparse autoencoder saved to {output_name}")
    
    print("\nNext steps:")
    print(f"1. Analyze features: bin/python analyze_sae_features.py --model {args.model} --checkpoint {args.checkpoint} --layer {args.layer}")
    print(f"2. Compare with other checkpoint: bin/python train_sparse_autoencoder.py --model {args.model} --checkpoint {'finetuned' if args.checkpoint == 'pretrained' else 'pretrained'}")


if __name__ == "__main__":
    main()
