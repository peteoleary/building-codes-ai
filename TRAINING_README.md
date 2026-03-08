# GPT-2 Fine-tuning on San Francisco Building Codes

This script fine-tunes a GPT-2 model on the San Francisco Building Code to create a specialized model that better understands construction regulations and building terminology.

## Overview

**train_san_francisco.py** takes a pre-trained GPT-2 model and fine-tunes it on the San Francisco Building Code text, creating a domain-specific model that can:
- Better understand building code language and terminology
- Generate more accurate responses to construction-related queries
- Provide more relevant information about San Francisco building regulations

## Requirements

### Python Environment
- Python 3.13+
- PyTorch 2.9+ (with MPS support for Apple Silicon or CUDA for NVIDIA GPUs)
- tiktoken
- Additional dependencies from the project

### Data Files
- `san_francisco-ca-1.txt` - San Francisco Building Code text file (must be in the same directory)

### Pre-trained Model
Choose one:
- `gpt2-small-124M.pth` (670 MB) - Faster training, smaller model
- `gpt2-medium-355M.pth` (1.6 GB) - Better quality, longer training

**Download the Medium model:**
```bash
bin/python download_gpt2_medium.py
```

## Quick Start

### 0. Setup Virtual Environment (First Time Only)

If you don't already have a virtual environment, create one:

```bash
# Create a new virtual environment in the current directory
python3 -m venv .

# Or specify Python 3.13+ explicitly
python3.13 -m venv .

# Activate the virtual environment
source bin/activate

# Verify Python version (should be 3.13+)
python --version

# Install required dependencies
pip install -r requirements.txt

# Verify installation
python check_dependencies.py
```

**Note:** The virtual environment creates `bin/`, `lib/`, and `include/` directories in your project folder. The `bin/` directory contains `python`, `pip`, and other executables.

### 1. Prepare Your Environment
```bash
# Activate virtual environment (if not already active)
source bin/activate

# Verify you have the required files
ls san_francisco-ca-1.txt  # Training data
ls gpt2-medium-355M.pth    # Pre-trained model
```

### 2. Run Training
```bash
bin/python train_san_francisco.py
```

The script will:
1. Load and tokenize the training data
2. Split into training (90%) and validation (10%) sets
3. Load the pre-trained GPT-2 model
4. Apply layer freezing for efficient fine-tuning
5. Train for 5 epochs with periodic evaluation
6. Save the fine-tuned model

### 3. Monitor Training
Watch for output like:
```
Ep 1/5 (Step 000050): Train loss 3.245, Val loss 3.198
Ep 1/5 (Step 000100): Train loss 2.891, Val loss 2.845
Ep 2/5 (Step 000150): Train loss 2.654, Val loss 2.612
```

**Good training:** Both losses decrease together  
**Overfitting:** Train loss decreases but val loss increases

## Configuration

### Model Configuration (Lines 14-22)

Currently set to **GPT-2 Medium (355M parameters)**:
```python
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,        # Medium: 1024, Small: 768
    "n_heads": 16,          # Medium: 16, Small: 12
    "n_layers": 24,         # Medium: 24, Small: 12
    "drop_rate": 0.1,
    "qkv_bias": True
}
```

### Training Hyperparameters (Lines 25-33)

```python
BATCH_SIZE = 2           # Reduced for larger model (use 4 for Small)
MAX_LENGTH = 256         # Sequence length for training
STRIDE = 128             # Sliding window stride
NUM_EPOCHS = 5           # Number of training epochs
LEARNING_RATE = 5e-5     # Conservative for fine-tuning
WEIGHT_DECAY = 0.1       # Regularization strength
EVAL_FREQ = 50           # Evaluate every N steps
EVAL_ITER = 5            # Batches to use for evaluation
```

### Layer Freezing Strategy (Line 268)

```python
setup_layer_freezing(model, strategy="freeze_early")
```

**Available strategies:**
- `"freeze_early"` (default) - Freeze 75% of layers (recommended)
- `"freeze_most"` - Only train last 2 layers + head (faster)
- `"head_only"` - Only train output head (fastest)
- `"no_freeze"` - Train everything (slower, more overfitting risk)

## Output Files

After training completes, you'll have:

### Fine-tuned Model
```
gpt2-medium-san-francisco-finetuned.pth  (~1.6 GB)
```
- Contains only the model weights
- Use this for inference and sharing

### Training Checkpoint
```
gpt2-medium-san-francisco-checkpoint.pth  (~3.2 GB)
```
- Contains model weights + optimizer state + training metrics
- Use this to resume training later

## Customization

### Training on Small Model (124M)

1. Change configuration (line 14):
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}
```

2. Update batch size (line 25):
```python
BATCH_SIZE = 4  # Can use larger batch with smaller model
```

3. Change model file (line 261):
```python
model_file = "gpt2-small-124M.pth"
```

4. Update references to config throughout

### Adjusting for Overfitting

If validation loss starts increasing:

1. **Increase dropout** (line 20):
```python
"drop_rate": 0.2,  # or 0.3
```

2. **Reduce epochs** (line 28):
```python
NUM_EPOCHS = 3  # or even 2
```

3. **Use more aggressive freezing** (line 268):
```python
setup_layer_freezing(model, strategy="freeze_most")
```

### Adjusting for Better Quality

If model isn't learning enough:

1. **Reduce learning rate** (line 29):
```python
LEARNING_RATE = 2e-5  # More conservative
```

2. **Train more layers** (line 268):
```python
setup_layer_freezing(model, strategy="no_freeze")
```

3. **Increase epochs** (line 28):
```python
NUM_EPOCHS = 10
```

## Training Time Estimates

On Apple M1/M2/M3 Mac:

| Model | Strategy | Batch Size | Time per Epoch | Total (5 epochs) |
|-------|----------|------------|----------------|------------------|
| Small (124M) | no_freeze | 4 | ~8-10 min | ~40-50 min |
| Small (124M) | freeze_early | 4 | ~5-7 min | ~25-35 min |
| Medium (355M) | no_freeze | 2 | ~15-20 min | ~75-100 min |
| Medium (355M) | freeze_early | 2 | ~10-12 min | ~50-60 min |

**Note:** Times vary based on hardware and system load.

## Memory Requirements

| Model | Configuration | Memory Usage |
|-------|--------------|--------------|
| Small (124M) | All layers | ~8-10 GB |
| Small (124M) | freeze_early | ~6-8 GB |
| Medium (355M) | All layers | ~16-20 GB |
| Medium (355M) | freeze_early | ~10-12 GB |

**Recommendation:** Use 16GB+ RAM/unified memory for Medium model.

## Understanding the Output

### During Training
```
Ep 1/5 (Step 000050): Train loss 3.245, Val loss 3.198
```
- **Ep**: Current epoch (1-5)
- **Step**: Training step number
- **Train loss**: Loss on training data (should decrease)
- **Val loss**: Loss on validation data (should decrease, stay close to train loss)

### Generated Samples
After each epoch, the model generates sample text:
```
Generated sample:
--------------------------------------------------------------------------------
The San Francisco Building Code requires that all residential buildings must
comply with fire safety regulations including smoke detectors...
--------------------------------------------------------------------------------
```
This helps you see how the model's generation quality improves.

### Final Test
At the end, you'll see responses to test prompts:
- "The San Francisco Building Code"
- "Section 101"
- "Building construction"

## Troubleshooting

### Out of Memory Error
**Solution:** Reduce batch size or use smaller model
```python
BATCH_SIZE = 1  # Minimum possible
```

### Training Too Slow
**Solution:** Use more aggressive layer freezing
```python
setup_layer_freezing(model, strategy="freeze_most")
```

### Model Not Learning (Loss Not Decreasing)
**Possible causes:**
- Learning rate too low → increase to 1e-4
- All parameters frozen → check layer freezing
- Data encoding issues → verify san_francisco-ca-1.txt is readable

### Loss Increases After Initial Decrease
**Cause:** Overfitting  
**Solution:** 
- Increase dropout rate
- Reduce number of epochs
- Use more aggressive layer freezing

## Next Steps

After training:

1. **Test the model** with [test_dual_models.py](test_dual_models.py):
```bash
bin/python test_dual_models.py
```

2. **Use in Chainlit app** - The fine-tuned model will be available in [app.py](app.py):
```bash
bin/chainlit run app.py --port 8000
```

3. **Compare models** - Switch between pretrained and fine-tuned in the app to see the difference

## Advanced Usage

### Resume Training from Checkpoint

```python
# Load checkpoint instead of pretrained model
checkpoint = torch.load("gpt2-medium-san-francisco-checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
previous_losses = checkpoint['train_losses']
```

### Fine-tune on Additional Data

1. Combine multiple text files:
```bash
cat san_francisco-ca-1.txt other-codes.txt > combined.txt
```

2. Update the script to load `combined.txt` instead

### Export for Production

The `.pth` files can be:
- Shared with team members
- Used in other PyTorch applications
- Loaded in the Chainlit app for inference

## Citation

Based on "Build a Large Language Model From Scratch" by Sebastian Raschka.

## License

See main project LICENSE file.
