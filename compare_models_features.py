"""
Compare Features Between Pretrained and Fine-tuned Models

Creates a side-by-side comparison showing which features are unique to
the fine-tuned model (indicating learned concepts from training data).

Usage:
    bin/python compare_models_features.py --model small --layer 6
"""

import torch
import argparse
from pathlib import Path
from train_sparse_autoencoder import SparseAutoencoder
import numpy as np


def load_autoencoder(model_size, checkpoint_type, layer_idx, device):
    """Load trained sparse autoencoder"""
    sae_path = f"sae_{model_size}_{checkpoint_type}_layer{layer_idx}.pth"
    
    if not Path(sae_path).exists():
        raise FileNotFoundError(
            f"Sparse autoencoder not found: {sae_path}\n"
            f"Train it first with:\n"
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
    
    autoencoder = SparseAutoencoder(input_dim, hidden_dim)
    autoencoder.load_state_dict(state_dict)
    autoencoder.to(device)
    autoencoder.eval()
    
    return autoencoder


def compute_feature_similarity(sae1, sae2):
    """
    Compute pairwise cosine similarity between features from two autoencoders
    
    Returns:
        similarity: (hidden_dim, hidden_dim) matrix of cosine similarities
        best_matches: For each feature in sae2, index of most similar feature in sae1
        max_similarities: Maximum similarity for each feature in sae2
    """
    # Get decoder weights (these are the feature directions)
    features1 = sae1.decoder.weight.data.T  # (hidden_dim, input_dim)
    features2 = sae2.decoder.weight.data.T
    
    # Normalize
    features1_norm = features1 / (features1.norm(dim=1, keepdim=True) + 1e-8)
    features2_norm = features2 / (features2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity
    similarity = torch.mm(features2_norm, features1_norm.T)  # (hidden_dim, hidden_dim)
    
    # Find best match for each feature in sae2
    max_similarities, best_matches = similarity.max(dim=1)
    
    return similarity, best_matches, max_similarities


def main():
    parser = argparse.ArgumentParser(description="Compare features between pretrained and finetuned models")
    parser.add_argument("--model", choices=["small", "medium"], required=True,
                       help="Model size")
    parser.add_argument("--layer", type=int, default=6,
                       help="Layer index (default: 6)")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold for novel features (default: 0.7)")
    
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load both sparse autoencoders
    print("Loading sparse autoencoders...")
    try:
        sae_pretrained = load_autoencoder(args.model, "pretrained", args.layer, device)
        print(f"✓ Loaded pretrained SAE")
        
        sae_finetuned = load_autoencoder(args.model, "finetuned", args.layer, device)
        print(f"✓ Loaded finetuned SAE")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Compute similarity
    print(f"\nComputing feature similarities...")
    similarity, best_matches, max_similarities = compute_feature_similarity(
        sae_pretrained, sae_finetuned
    )
    
    # Find novel features (low similarity to any pretrained feature)
    novel_mask = max_similarities < args.threshold
    novel_features = novel_mask.nonzero().squeeze()
    
    # Find preserved features (high similarity to a pretrained feature)
    preserved_mask = max_similarities >= args.threshold
    preserved_features = preserved_mask.nonzero().squeeze()
    
    # Statistics
    num_novel = novel_features.numel()
    num_preserved = preserved_features.numel()
    total_features = sae_finetuned.hidden_dim
    
    print("\n" + "="*80)
    print("FEATURE COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nTotal features: {total_features}")
    print(f"Novel features (similarity < {args.threshold}): {num_novel} ({100*num_novel/total_features:.1f}%)")
    print(f"Preserved features (similarity ≥ {args.threshold}): {num_preserved} ({100*num_preserved/total_features:.1f}%)")
    
    # Show most novel features
    if num_novel > 0:
        print("\n" + "─"*80)
        print("MOST NOVEL FEATURES (likely learned from SF building codes)")
        print("─"*80)
        
        # Sort by increasing similarity (most novel first)
        novel_sorted_indices = torch.argsort(max_similarities[novel_mask])
        novel_sorted_features = novel_features[novel_sorted_indices]
        novel_sorted_similarities = max_similarities[novel_mask][novel_sorted_indices]
        
        num_to_show = min(20, num_novel)
        print(f"\nTop {num_to_show} most novel features:\n")
        print(f"{'Rank':<6} {'Feature ID':<12} {'Best Match Similarity':<25} {'Novelty'}")
        print("─"*80)
        
        for i in range(num_to_show):
            feat_idx = novel_sorted_features[i].item()
            similarity_val = novel_sorted_similarities[i].item()
            novelty = 1 - similarity_val
            
            similarity_bar = "█" * int(similarity_val * 20)
            novelty_bar = "█" * int(novelty * 20)
            
            print(f"{i+1:<6} {feat_idx:<12} {similarity_val:.3f} {similarity_bar:<20} {novelty_bar}")
    
    # Show most preserved features
    if num_preserved > 0:
        print("\n" + "─"*80)
        print("MOST PRESERVED FEATURES (similar to pretrained)")
        print("─"*80)
        
        # Sort by decreasing similarity (most preserved first)
        preserved_sorted_indices = torch.argsort(max_similarities[preserved_mask], descending=True)
        preserved_sorted_features = preserved_features[preserved_sorted_indices]
        preserved_sorted_similarities = max_similarities[preserved_mask][preserved_sorted_indices]
        
        num_to_show = min(10, num_preserved)
        print(f"\nTop {num_to_show} most preserved features:\n")
        print(f"{'Rank':<6} {'Feature ID':<12} {'Similarity':<25}")
        print("─"*80)
        
        for i in range(num_to_show):
            feat_idx = preserved_sorted_features[i].item()
            similarity_val = preserved_sorted_similarities[i].item()
            
            similarity_bar = "█" * int(similarity_val * 20)
            
            print(f"{i+1:<6} {feat_idx:<12} {similarity_val:.3f} {similarity_bar}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    print(f"""
Novel features ({num_novel}):
  These features are unique to the fine-tuned model. They likely represent:
  - Building code terminology and concepts
  - San Francisco-specific construction regulations
  - Technical language patterns from the training data
  - Regulatory structures and formats

Preserved features ({num_preserved}):
  These features are similar between models. They likely represent:
  - General language patterns
  - Common words and phrases
  - Basic grammatical structures
  - General semantic concepts

Next steps:
  1. Analyze novel features to see what they activate on:
     bin/python analyze_sae_features.py --model {args.model} --checkpoint finetuned --layer {args.layer}
  
  2. Try different layers to see how features evolve:
     bin/python compare_models_features.py --model {args.model} --layer {args.layer-1}
     bin/python compare_models_features.py --model {args.model} --layer {args.layer+1}
  
  3. Adjust the similarity threshold to see more/fewer novel features:
     bin/python compare_models_features.py --model {args.model} --threshold 0.8
""")


if __name__ == "__main__":
    main()
