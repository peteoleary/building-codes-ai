# GPT-2 Fine-tuning on San Francisco Building Codes

This script fine-tunes both GPT-2 Small and Medium models on the San Francisco Building Code to create specialized models that better understand construction regulations and building terminology.

## Overview

**train_san_francisco.py** trains both GPT-2 Small (124M) and Medium (355M) models sequentially on the San Francisco Building Code text, creating domain-specific models that can:
- Better understand building code language and terminology
- Generate more accurate responses to construction-related queries
- Provide more relevant information about San Francisco building regulations

**Output:** 4 model files (2 fine-tuned + 2 checkpoints) for use in the Chainlit app

## Requirements

### Python Environment
- Python 3.13+
- PyTorch 2.9+ (with MPS support for Apple Silicon or CUDA for NVIDIA GPUs)
- tiktoken
- Additional dependencies from the project

### Data Files
- `san_francisco-ca-1.txt` - San Francisco Building Code text file (must be in the same directory)

### Pre-trained Models
Both required (script trains both models):
- `gpt2-small-124M.pth` (670 MB) - Small model, faster training
- `gpt2-medium-355M.pth` (1.6 GB) - Medium model, better quality

**Download both models:**
```bash
bin/python download_pretrained_models.py
```

This script downloads both Small (124M) and Medium (355M) models. If you already have one, it will skip it and only download the missing one.

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

#### Upgrading Existing Virtual Environment to Python 3.13

If you have an older virtual environment (e.g., Python 3.11 or 3.12), you need to recreate it with Python 3.13:

```bash
# 1. Check your current Python version
python --version

# 2. If not Python 3.13+, deactivate the current environment
deactivate

# 3. Remove the old virtual environment
# WARNING: This deletes bin/, lib/, and include/ directories
rm -rf bin lib include pyvenv.cfg

# 4. Install Python 3.13 if needed
```

**Installing Python 3.13 by operating system:**

**macOS (Homebrew):**
```bash
brew install python@3.13
```

**Ubuntu/Debian:**
```bash
# Add deadsnakes PPA for Python 3.13
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev

# Verify installation
python3.13 --version
```

**Fedora/RHEL/CentOS:**
```bash
# Fedora 39+ should have Python 3.13 in repos
sudo dnf install python3.13 python3.13-devel

# For older versions, build from source or use pyenv (see below)
```

**Arch Linux:**
```bash
# Python 3.13 should be in official repos
sudo pacman -S python

# Or explicitly install 3.13 if available
sudo pacman -S python313
```

**Using pyenv (all Linux distributions):**
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Add to ~/.bashrc or ~/.zshrc:
# export PATH="$HOME/.pyenv/bin:$PATH"
# eval "$(pyenv init -)"

# Restart shell, then install Python 3.13
pyenv install 3.13.7
pyenv local 3.13.7

# Verify
python --version
```

**Building from source (any Linux):**
```bash
# Install build dependencies (Ubuntu/Debian example)
sudo apt install build-essential zlib1g-dev libncurses5-dev \
  libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
  libffi-dev libsqlite3-dev wget libbz2-dev

# Download and build Python 3.13
cd /tmp
wget https://www.python.org/ftp/python/3.13.7/Python-3.13.7.tgz
tar -xf Python-3.13.7.tgz
cd Python-3.13.7
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall  # Use altinstall to not override system python

# Verify
python3.13 --version
```

**Continue with virtual environment setup:**

```bash
# 5. Create new virtual environment with Python 3.13
python3.13 -m venv .

# 6. Activate the new environment
source bin/activate

# 7. Verify Python version
python --version  # Should show Python 3.13.x

# 8. Reinstall dependencies
pip install -r requirements.txt

# 9. Verify installation
python check_dependencies.py
```

**Important Notes:**
- **Why Python 3.13?** PyTorch 2.9+ with MPS (Apple Silicon) and latest CUDA support requires Python 3.13+
- **Older Python versions:** If stuck on Python 3.11/3.12, you can try:
  - Installing PyTorch 2.5-2.8 (may work but without latest features)
  - Using CPU-only mode (no MPS/CUDA acceleration)
  - Using Docker with Python 3.13 pre-installed
- **System Python:** Avoid using system Python for this project; always use a virtual environment to prevent conflicts
- **Multiple Python versions:** Tools like `pyenv` let you have multiple Python versions side-by-side without conflicts

**Troubleshooting:**
- If `python3.13` command not found after installation, try `python3` or check your PATH
- On some systems, you may need to use `python3.13-venv` instead of `-m venv`
- If build fails on Linux, ensure all development headers are installed (`-dev` or `-devel` packages)

### 1. Prepare Your Environment
```bash
# Activate virtual environment (if not already active)
source bin/activate

# Verify you have the required files
ls san_francisco-ca-1.txt   # Training data
ls gpt2-small-124M.pth      # Small pre-trained model
ls gpt2-medium-355M.pth     # Medium pre-trained model
```

### 2. Run Training
```bash
bin/python train_san_francisco.py
```

The script will:
1. Load and tokenize the training data
2. Split into training (90%) and validation (10%) sets
3. **Train GPT-2 Small (124M):**
   - Load pre-trained Small model
   - Apply layer freezing (freeze 75% of layers)
   - Train for 5 epochs with batch_size=4
   - Save fine-tuned model and checkpoint
4. **Train GPT-2 Medium (355M):**
   - Load pre-trained Medium model
   - Apply layer freezing (freeze 75% of layers)
   - Train for 5 epochs with batch_size=2
   - Save fine-tuned model and checkpoint
5. Display training summary with times for both models

**Total Runtime:** ~70-85 minutes (Small: ~25 min, Medium: ~50 min)

### 3. Monitor Training

The script trains both models sequentially. You'll see:

**1. GPT-2 Small (124M) training:**
```
================================================================================
TRAINING GPT-2 Small (124M)
================================================================================
Loading pre-trained model from gpt2-small-124M.pth...
✓ Pre-trained model loaded successfully

Setting up layer freezing...
✓ Frozen first 9/12 transformer blocks
...
Ep 1/5 (Step 000050): Train loss 3.245, Val loss 3.198
Ep 1/5 (Step 000100): Train loss 2.891, Val loss 2.845
...
Training completed in 25.43 minutes
✓ GPT-2 Small (124M) training complete!
```

**2. GPT-2 Medium (355M) training:**
```
================================================================================
TRAINING GPT-2 Medium (355M)
================================================================================
Loading pre-trained model from gpt2-medium-355M.pth...
✓ Pre-trained model loaded successfully

Setting up layer freezing...
✓ Frozen first 18/24 transformer blocks
...
Ep 1/5 (Step 000050): Train loss 3.112, Val loss 3.089
...
Training completed in 52.18 minutes
✓ GPT-2 Medium (355M) training complete!
```

**3. Final summary:**
```
================================================================================
ALL TRAINING COMPLETE!
================================================================================

Total training time: 77.61 minutes

Individual model times:
  GPT-2 Small (124M): 25.43 minutes
  GPT-2 Medium (355M): 52.18 minutes

Fine-tuned models ready for use:
  ✓ gpt2-san-francisco-finetuned.pth
  ✓ gpt2-medium-san-francisco-finetuned.pth
================================================================================
```

**Good training:** Both losses decrease together  
**Overfitting:** Train loss decreases but val loss increases

## Configuration

### Model Configurations (Lines 14-51)

The script trains both models using these configurations:

**GPT-2 Small (124M parameters):**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,         # Small: 768
    "n_heads": 12,          # Small: 12
    "n_layers": 12,         # Small: 12
    "drop_rate": 0.1,
    "qkv_bias": True
}
```

**GPT-2 Medium (355M parameters):**
```python
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,        # Medium: 1024
    "n_heads": 16,          # Medium: 16
    "n_layers": 24,         # Medium: 24
    "drop_rate": 0.1,
    "qkv_bias": True
}
```

**Model-specific settings:**
- Small: batch_size=4, output: `gpt2-san-francisco-finetuned.pth`
- Medium: batch_size=2, output: `gpt2-medium-san-francisco-finetuned.pth`

### Training Hyperparameters (Lines 54-61)

```python
MAX_LENGTH = 256         # Sequence length for training
STRIDE = 128             # Sliding window stride
NUM_EPOCHS = 5           # Number of training epochs
LEARNING_RATE = 5e-5     # Conservative for fine-tuning
WEIGHT_DECAY = 0.1       # Regularization strength
EVAL_FREQ = 50           # Evaluate every N steps
EVAL_ITER = 5            # Batches to use for evaluation
```

### Layer Freezing Strategy

```python
setup_layer_freezing(model, strategy="freeze_early")
```

**Available strategies:**
- `"freeze_early"` (default) - Freeze 75% of layers (recommended)
- `"freeze_most"` - Only train last 2 layers + head (faster)
- `"head_only"` - Only train output head (fastest)
- `"no_freeze"` - Train everything (slower, more overfitting risk)

## Output Files

After training completes, you'll have **4 files**:

### Small Model Files
```
gpt2-san-francisco-finetuned.pth  (~670 MB)
```
- Contains Small model weights only
- Use for inference (faster, less memory)

```
gpt2-small-san-francisco-checkpoint.pth  (~1.4 GB)
```
- Contains Small model + optimizer + training metrics
- Use to resume training

### Medium Model Files
```
gpt2-medium-san-francisco-finetuned.pth  (~1.6 GB)
```
- Contains Medium model weights only
- Use for inference (better quality)

```
gpt2-medium-san-francisco-checkpoint.pth  (~3.2 GB)
```
- Contains Medium model + optimizer + training metrics
- Use to resume training

**Total disk space needed:** ~7 GB for all output files

## Customization

### Training Only One Model

To train only Small or only Medium, edit the main loop (line ~390):

**Small only:**
```python
for model_key in ["small"]:
```

**Medium only:**
```python
for model_key in ["medium"]:
```

### Adjusting Batch Sizes

Edit MODEL_CONFIGS (lines 34-51) to change batch sizes:
```python
"small": {
    "batch_size": 8,  # Increase if you have more memory
    ...
},
"medium": {
    "batch_size": 4,  # Default is 2
    ...
}
```

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

### Script Total Time (Both Models)

Since the script trains both models sequentially:
- **Small (freeze_early) + Medium (freeze_early): ~70-85 minutes**
- **Small (no_freeze) + Medium (no_freeze): ~115-150 minutes**

**Note:** Times vary based on hardware and system load.

## Memory Requirements

| Model | Configuration | Memory Usage |
|-------|--------------|--------------|
| Small (124M) | All layers | ~8-10 GB |
| Small (124M) | freeze_early | ~6-8 GB |
| Medium (355M) | All layers | ~16-20 GB |
| Medium (355M) | freeze_early | ~10-12 GB |

**Peak memory:** Medium model (script runs models sequentially, not simultaneously)

**Recommendation:** 
- 16GB+ RAM/unified memory for both models
- 8GB may work for Small model only (edit script to skip Medium)
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

After training completes:

1. **Test both models** with [test_dual_models.py](test_dual_models.py):
```bash
bin/python test_dual_models.py
```

2. **Use in Chainlit app** - All 4 models will be available in [app.py](app.py):
```bash
bin/chainlit run app.py --port 8000
```

3. **Compare models** - Use the dropdown menu to switch between:
   - GPT-2 Small (Pretrained)
   - GPT-2 Small (San Francisco Fine-tuned) ← **New**
   - GPT-2 Medium (Pretrained)
   - GPT-2 Medium (San Francisco Fine-tuned) ← **New**

4. **Evaluate improvements** - Ask building code questions and compare responses across all 4 models

## Advanced Usage

### Resume Training from Checkpoint

Load either checkpoint to continue training:

**Small model:**
```python
checkpoint = torch.load("gpt2-small-san-francisco-checkpoint.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
previous_losses = checkpoint['train_losses']
```

**Medium model:**
```python
checkpoint = torch.load("gpt2-medium-san-francisco-checkpoint.pth")
model = GPTModel(GPT_CONFIG_355M)
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
