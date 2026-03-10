"""
Analyze Sparse Autoencoder Features

This script analyzes the features learned by sparse autoencoders,
finding the most activating text examples for each feature.

Usage:
    bin/python analyze_sae_features.py --model small --checkpoint pretrained --layer 6
    bin/python analyze_sae_features.py --model small --checkpoint finetuned --layer 6
"""

import torch
import tiktoken
import argparse
from pathlib import Path
from gpt_model import GPTModel
import numpy as np
from collections import defaultdict
from train_sparse_autoencoder import SparseAutoencoder, GPT_CONFIG_124M, GPT_CONFIG_355M

def load_autoencoder(model_size, checkpoint_type, layer_idx, device):
    """Load trained sparse autoencoder"""
    sae_path = f"sae_{model_size}_{checkpoint_type}_layer{layer_idx}.pth"
    
    if not Path(sae_path).exists():
        raise FileNotFoundError(
            f"Sparse autoencoder not found: {sae_path}\n"
            f"Please train it first with:\n"
            f"  bin/python train_sparse_autoencoder.py --model {model_size} --checkpoint {checkpoint_type} --layer {layer_idx}"
        )
    
    # Load checkpoint and infer dimensions
    checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    
    # Handle both old (state_dict only) and new (metadata) formats
    if 'encoder.weight' in checkpoint:
        # Old format: state_dict directly
        hidden_dim, input_dim = checkpoint['encoder.weight'].shape
        state_dict = checkpoint
    elif 'model_state_dict' in checkpoint:
        # New format: checkpoint with metadata
        hidden_dim, input_dim = checkpoint['model_state_dict']['encoder.weight'].shape
        state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError(f"Invalid checkpoint format: {sae_path}")
    
    # Load autoencoder
    autoencoder = SparseAutoencoder(input_dim, hidden_dim)
    autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)
    autoencoder.eval()
    
    return autoencoder


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
    
    print(f"Loading {model_size} {checkpoint_type} model...")
    model = GPTModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, config


def analyze_features(model, autoencoder, layer_idx, num_features=20, device='cpu'):
    """
    Find the most activating text examples for top features
    """
    print(f"\nAnalyzing top {num_features} features...")
    
    # Load tokenizer and text
    tokenizer = tiktoken.get_encoding("gpt2")
    data_file = "san_francisco-ca-1.txt"
    with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    
    # Collect activations and track feature activations
    max_length = 256
    feature_activations = defaultdict(list)  # feature_idx -> list of (activation, token_pos, text)
    
    # Hook to capture activations
    captured_acts = []
    
    def hook_fn(module, input, output):
        captured_acts.append(output.detach())
    
    hook = model.trf_blocks[layer_idx].register_forward_hook(hook_fn)
    
    print("Scanning text for feature activations...")
    with torch.no_grad():
        for i in range(0, len(tokens) - max_length, max_length // 2):  # Sliding window
            # Get token sequence
            token_seq = tokens[i:i + max_length]
            input_ids = torch.tensor([token_seq], dtype=torch.long).to(device)
            
            # Forward pass through model
            captured_acts.clear()
            _ = model(input_ids)
            
            if captured_acts:
                acts = captured_acts[0]  # (1, seq_len, emb_dim)
                
                # Encode through sparse autoencoder
                acts_flat = acts.reshape(-1, acts.shape[-1])  # (seq_len, emb_dim)
                features = autoencoder.encode(acts_flat)  # (seq_len, hidden_dim)
                
                # For each position, find top activating features
                for pos in range(features.shape[0]):
                    feature_vals = features[pos]
                    
                    # Get token and surrounding context
                    token_idx = i + pos
                    if token_idx >= len(tokens):
                        continue
                    
                    # Get text context (5 tokens before and after)
                    context_start = max(0, token_idx - 5)
                    context_end = min(len(tokens), token_idx + 6)
                    context_tokens = tokens[context_start:context_end]
                    context_text = tokenizer.decode(context_tokens)
                    
                    # Record top feature activations for this position
                    top_k = 5
                    top_features = torch.topk(feature_vals, k=top_k)
                    
                    for feat_idx, feat_val in zip(top_features.indices, top_features.values):
                        if feat_val > 0:  # Only record positive activations
                            feature_activations[feat_idx.item()].append({
                                'activation': feat_val.item(),
                                'token_pos': token_idx,
                                'context': context_text,
                                'token': tokenizer.decode([tokens[token_idx]])
                            })
    
    hook.remove()
    
    # Find most active features
    feature_max_activations = {
        feat_idx: max(acts, key=lambda x: x['activation'])['activation']
        for feat_idx, acts in feature_activations.items()
    }
    
    top_features = sorted(feature_max_activations.items(), key=lambda x: x[1], reverse=True)[:num_features]
    
    print(f"\n{'='*80}")
    print(f"TOP {num_features} FEATURES")
    print(f"{'='*80}\n")
    
    for rank, (feat_idx, max_activation) in enumerate(top_features, 1):
        activations = feature_activations[feat_idx]
        
        # Sort by activation strength
        activations_sorted = sorted(activations, key=lambda x: x['activation'], reverse=True)
        
        print(f"\n{'─'*80}")
        print(f"Feature {feat_idx} (Rank #{rank})")
        print(f"Max activation: {max_activation:.3f}")
        print(f"Number of activations: {len(activations)}")
        print(f"{'─'*80}")
        
        # Show top 5 activating examples
        print("\nTop activating examples:")
        for i, act_info in enumerate(activations_sorted[:5], 1):
            print(f"\n  {i}. Activation: {act_info['activation']:.3f}")
            print(f"     Token: '{act_info['token']}'")
            print(f"     Context: {act_info['context']}")
    
    return top_features, feature_activations


def compare_features(model_size, layer_idx, device):
    """Compare features between pretrained and finetuned models"""
    print("\n" + "="*80)
    print("COMPARING PRETRAINED VS FINETUNED FEATURES")
    print("="*80)
    
    # Load both autoencoders
    print("\nLoading pretrained sparse autoencoder...")
    sae_pretrained = load_autoencoder(model_size, "pretrained", layer_idx, device)
    
    print("Loading finetuned sparse autoencoder...")
    sae_finetuned = load_autoencoder(model_size, "finetuned", layer_idx, device)
    
    # Compare decoder weights (these represent the feature directions)
    pretrained_features = sae_pretrained.decoder.weight.data  # (input_dim, hidden_dim)
    finetuned_features = sae_finetuned.decoder.weight.data
    
    # Normalize for comparison
    pretrained_norm = pretrained_features / (pretrained_features.norm(dim=0, keepdim=True) + 1e-8)
    finetuned_norm = finetuned_features / (finetuned_features.norm(dim=0, keepdim=True) + 1e-8)
    
    # Compute cosine similarity matrix
    similarity = torch.mm(finetuned_norm.T, pretrained_norm)  # (hidden_dim, hidden_dim)
    
    # Find features that changed the most
    max_similarities, best_matches = similarity.max(dim=1)
    
    # Features with low similarity to any pretrained feature are "new"
    new_feature_threshold = 0.7
    new_features = (max_similarities < new_feature_threshold).nonzero().squeeze()
    
    print(f"\nFound {len(new_features)} potentially novel features in finetuned model")
    print(f"(cosine similarity < {new_feature_threshold} with any pretrained feature)")
    
    if len(new_features) > 0:
        print(f"\nTop 10 most novel features:")
        novel_sorted = sorted(
            [(idx.item(), max_similarities[idx].item()) for idx in new_features],
            key=lambda x: x[1]
        )[:10]
        
        for rank, (feat_idx, similarity) in enumerate(novel_sorted, 1):
            print(f"  {rank}. Feature {feat_idx}: similarity = {similarity:.3f}")
    
    return new_features


def main():
    parser = argparse.ArgumentParser(description="Analyze sparse autoencoder features")
    parser.add_argument("--model", choices=["small", "medium"], required=True,
                       help="Model size")
    parser.add_argument("--checkpoint", choices=["pretrained", "finetuned"], required=True,
                       help="Checkpoint type")
    parser.add_argument("--layer", type=int, default=6,
                       help="Layer index (default: 6)")
    parser.add_argument("--num_features", type=int, default=20,
                       help="Number of top features to analyze (default: 20)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare pretrained vs finetuned features")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load sparse autoencoder
    print(f"\nLoading sparse autoencoder for {args.model} {args.checkpoint} (layer {args.layer})...")
    autoencoder = load_autoencoder(args.model, args.checkpoint, args.layer, device)
    print(f"✓ Loaded sparse autoencoder with {autoencoder.hidden_dim} features")
    
    # Load model
    model, config = load_model(args.model, args.checkpoint, device)
    print("✓ Model loaded")
    
    # Analyze features
    top_features, feature_activations = analyze_features(
        model, autoencoder, args.layer,
        num_features=args.num_features,
        device=device
    )
    
    # Compare if requested
    if args.compare:
        try:
            new_features = compare_features(args.model, args.layer, device)
        except FileNotFoundError as e:
            print(f"\n⚠ Cannot compare: {e}")
            print("Train both pretrained and finetuned sparse autoencoders first.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nThese features represent what the model 'pays attention to' internally.")
    print("Finetuned features should show building-code-specific patterns!")


if __name__ == "__main__":
    main()
