# Sparse Autoencoder Analysis for GPT-2

Discover interpretable features in your GPT-2 models using sparse autoencoders, inspired by OpenAI and Anthropic's research.

## What Are Sparse Autoencoders?

Sparse autoencoders help us understand what neural networks "pay attention to" by decomposing their internal activations into interpretable features. This lets you:

- **Discover what your model learned** from fine-tuning on SF building codes
- **Find building-code-specific features** (like Anthropic found a "Golden Gate Bridge" feature)
- **Compare pretrained vs fine-tuned models** to see what changed
- **Visualize model internals** in a human-understandable way

## Quick Start

### 1. Train Sparse Autoencoders

Train on your pretrained model:
```bash
bin/python train_sparse_autoencoder.py --model small --checkpoint pretrained
```

Train on your fine-tuned model:
```bash
bin/python train_sparse_autoencoder.py --model small --checkpoint finetuned
```

**Options:**
- `--model`: `small` (124M) or `medium` (355M)
- `--checkpoint`: `pretrained` or `finetuned`
- `--layer`: Which layer to analyze (default: 6, middle layer)
- `--hidden_dim`: Number of features to learn (default: 8192)
- `--epochs`: Training epochs (default: 10)

**Training time:** ~10-20 minutes on M1/M2/M3 Mac

### 2. Analyze Features

Find the most important features:
```bash
bin/python analyze_sae_features.py --model small --checkpoint finetuned --num_features 20
```

Compare pretrained vs finetuned:
```bash
bin/python analyze_sae_features.py --model small --checkpoint finetuned --compare
```

## Example Workflow

```bash
# 1. First, fine-tune your models (if not already done)
bin/python train_san_francisco.py

# 2. Train sparse autoencoders on pretrained model
bin/python train_sparse_autoencoder.py --model small --checkpoint pretrained --layer 6

# 3. Train sparse autoencoders on finetuned model
bin/python train_sparse_autoencoder.py --model small --checkpoint finetuned --layer 6

# 4. Analyze finetuned features
bin/python analyze_sae_features.py --model small --checkpoint finetuned --num_features 20

# 5. Compare pretrained vs finetuned
bin/python analyze_sae_features.py --model small --checkpoint finetuned --compare
```

## What to Look For

### In Pretrained Models
You might find features for:
- Common words and phrases
- Grammatical patterns
- General semantic concepts

### In Fine-tuned Models
You should find NEW features for:
- **Building codes terminology** ("Section 101", "fire safety", "egress")
- **San Francisco specific terms** ("SFBC", "DBI", San Francisco proper nouns)
- **Technical construction terms** ("structural", "load-bearing", "occupancy")
- **Regulatory language patterns** ("shall", "must comply", "except as")

### Example Output

```
================================================================================
TOP 20 FEATURES
================================================================================

────────────────────────────────────────────────────────────────────────────────
Feature 1547 (Rank #1)
Max activation: 12.453
Number of activations: 234
────────────────────────────────────────────────────────────────────────────────

Top activating examples:

  1. Activation: 12.453
     Token: 'Section'
     Context: ...Building Code Section 101 General Provisions...

  2. Activation: 11.892
     Token: 'Section'
     Context: ...Section 3403 shall not apply to buildings...

  3. Activation: 10.334
     Token: 'shall'
     Context: ...occupancy load shall not exceed the maximum...
```

## Understanding the Results

**Feature Rank**: Higher rank = more frequently/strongly activated  
**Max Activation**: How strongly the feature fires at its peak  
**Context**: Text where the feature activates most

**Novel features** (in comparison mode): Features present in fine-tuned but not pretrained model, indicating learned concepts from your  training data.

## Technical Details

### Architecture

```
Input: Model activations (768D for Small, 1024D for Medium)
       ↓
Encoder: Linear + ReLU
       ↓
Sparse Features: 8192D (by default)
       ↓
Decoder: Linear
       ↓
Output: Reconstructed activations
```

**Loss Function**: MSE reconstruction + L1 sparsity penalty

### Which Layer to Analyze?

- **Early layers (0-4)**: More syntactic features (punctuation, word boundaries)
- **Middle layers (5-8)**: Balanced semantic and syntactic features ← **Recommended**
- **Late layers (9+)**: More task-specific features

Try layer 6 first (middle of Small model), then experiment with others.

### Computational Requirements

| Task | Model | Time (M1/M2/M3) | Memory |
|------|-------|-----------------|---------|
| Collect activations | Small | ~5 min | ~4 GB |
| Collect activations | Medium | ~8 min | ~6 GB |
| Train SAE | Small | ~10 min | ~6 GB |
| Train SAE | Medium | ~15 min | ~8 GB |
| Analyze features | Either | ~5 min | ~4 GB |

## Advanced Usage

### Multiple Layers

Analyze different layers to see how features evolve:

```bash
for layer in 3 6 9; do
    bin/python train_sparse_autoencoder.py --model small --checkpoint finetuned --layer $layer
    bin/python analyze_sae_features.py --model small --checkpoint finetuned --layer $layer
done
```

### Larger Feature Sets

Train with more features to capture rarer concepts:

```bash
bin/python train_sparse_autoencoder.py \
    --model small \
    --checkpoint finetuned \
    --hidden_dim 16384 \
    --epochs 15
```

**Note**: Larger hidden_dim = more features but slower training and more memory

### Adjust Sparsity

Control how sparse features are with `--l1_coeff`:

- `--l1_coeff 1e-4`: Less sparse, more active features
- `--l1_coeff 1e-3`: Default, balanced
- `--l1_coeff 1e-2`: Very sparse, fewer active features

## Related Research

- **OpenAI (2022)**: [Sparse Autoencoders for GPT-2](https://github.com/openai/sparse_autoencoder)
- **Anthropic (2024)**: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
  - Found interpretable features in Claude 3 Sonnet
  - Golden Gate Bridge feature that can control model behavior
  - Safety-relevant features for bias, deception, etc.

## Troubleshooting

### "Out of memory" error
- Reduce `--hidden_dim` (try 4096)
- Reduce `--num_samples` (try 5000)
- Use CPU instead of MPS: `export PYTORCH_ENABLE_MPS_FALLBACK=1`

### Features seem random/uninterpretable
- Try a different layer (`--layer`)
- Increase training epochs (`--epochs 20`)
- Adjust sparsity (`--l1_coeff`)

### Training is too slow
- Reduce `--num_samples` (default: 10000)
- Reduce `--hidden_dim` (default: 8192)
- Use smaller model (`--model small`)

## Output Files

- `sae_small_pretrained_layer6.pth`: Trained autoencoder for pretrained Small model
- `sae_small_finetuned_layer6.pth`: Trained autoencoder for fine-tuned Small model
- `sae_medium_pretrained_layer6.pth`: Trained autoencoder for pretrained Medium model
- `sae_medium_finetuned_layer6.pth`: Trained autoencoder for fine-tuned Medium model

**Storage**: ~200-500 MB per trained autoencoder

## Next Steps

Once you've identified interesting features:

1. **Feature steering**: Modify model behavior by clamping features (like the Golden Gate Bridge example)
2. **Ablation studies**: Remove features to see their causal effect
3. **Circuit analysis**: Trace how features connect across layers
4. **Safety analysis**: Look for bias, deception, or problematic features

## Questions?

This is cutting-edge interpretability research! The field is evolving rapidly. Your results will help understand how domain-specific fine-tuning changes model internals.
