# GPT-2 Model Comparison App

This Chainlit app allows you to compare responses from multiple GPT-2 models:
1. **GPT-2 Small (Pretrained)** - 124M parameter pretrained model
2. **GPT-2 Small (San Francisco Fine-tuned)** - 124M model fine-tuned on SF building codes
3. **GPT-2 Medium (Pretrained)** - 355M parameter pretrained model
4. **GPT-2 Medium (San Francisco Fine-tuned)** - 355M model fine-tuned on SF building codes

## How to Use

### Starting the App

Run the app using:
```bash
bin/chainlit run app.py --port 8000
```

Or use the VS Code launch configuration "Chainlit: Run App"

### Switching Between Models

1. Click the **settings icon** (⚙️) in the chat interface
2. Select your desired model from the **"Select Model"** dropdown
3. The app will confirm the switch with a message
4. All subsequent messages will use the selected model

### Available Models

**Small Models (124M parameters):**
- **GPT-2 Small (Pretrained)**: General-purpose model trained on internet text
  - Good for general conversation and text generation
  - Broad knowledge across many topics
  - Faster inference
  
- **GPT-2 Small (San Francisco Fine-tuned)**: Specialized model trained on SF building codes
  - Better understanding of building regulations and construction terminology
  - More relevant for San Francisco-specific building code questions
  - Faster inference

**Medium Models (355M parameters):**
- **GPT-2 Medium (Pretrained)**: Larger general-purpose model
  - Better language understanding and generation quality
  - More coherent and detailed responses
  - Slower inference than Small
  
- **GPT-2 Medium (San Francisco Fine-tuned)**: Larger specialized model
  - Best understanding of SF building codes and construction
  - Most detailed and accurate responses for building code questions
  - Slower inference than Small

## Testing the Difference

Try asking the same question to different models to see how model size and fine-tuning affect responses:

**Example prompts:**
- "What is the correct riser height for stairs in San Francisco?"
- "Section 101 of the building code"
- "Fire safety requirements for residential buildings"

**Expected behavior:**
- **Small Pretrained**: General knowledge, may not be specific to SF codes
- **Small Fine-tuned**: More specific to SF building codes, faster inference
- **Medium Pretrained**: Better general responses, more coherent
- **Medium Fine-tuned**: Best SF code responses, most detailed and accurate

## Requirements

- Python 3.13+
- PyTorch with MPS support (for Apple Silicon) or CUDA
- Chainlit
- tiktoken
- Model files (at least one of the following):
  - `gpt2-small-124M.pth` (670 MB)
  - `gpt2-san-francisco-finetuned.pth` (670 MB)
  - `gpt2-medium-355M.pth` (1.6 GB)
  - `gpt2-medium-san-francisco-finetuned.pth` (1.6 GB)

**Note**: The app will load only the models that are present in the directory.

## Model Details

### Small Models (124M parameters)
- Architecture: GPT-2 Small
- Total parameters: 124,439,808
- Context length: 1024 tokens
- Embedding dimension: 768
- Attention heads: 12
- Layers: 12
- Model file size: ~670 MB

### Medium Models (355M parameters)
- Architecture: GPT-2 Medium
- Total parameters: 406,286,336
- Context length: 1024 tokens
- Embedding dimension: 1024
- Attention heads: 16
- Layers: 24
- Model file size: ~1.6 GB

### Common Settings
- Tokenizer: GPT-2 BPE
- Vocabulary size: 50,257 tokens

## Generation Settings

- Temperature: 0.7
- Top-k sampling: 50
- Max new tokens: 150
- End of sequence token: 50256 (GPT-2 EOS)

## Performance Notes

**Inference Speed:**
- Small models (124M): ~2-3 seconds per response
- Medium models (355M): ~5-7 seconds per response

**Memory Usage:**
- Small models: ~2-3 GB VRAM/RAM
- Medium models: ~7-8 GB VRAM/RAM

**Recommendation:**
- For quick testing and general queries: Use Small models
- For detailed SF building code queries: Use Medium Fine-tuned model
- For comparison: Test the same prompt across all available models

## Creating Fine-tuned Models

The two fine-tuned models can be created by running the training script:

```bash
bin/python train_san_francisco.py
```

This script will:
1. Train **GPT-2 Small (124M)** on San Francisco building codes (~25 minutes)
   - Creates `gpt2-san-francisco-finetuned.pth`
   - Creates `gpt2-small-san-francisco-checkpoint.pth` (for resuming training)

2. Train **GPT-2 Medium (355M)** on San Francisco building codes (~50 minutes)
   - Creates `gpt2-medium-san-francisco-finetuned.pth`
   - Creates `gpt2-medium-san-francisco-checkpoint.pth` (for resuming training)

**Total training time:** ~70-85 minutes for both models

After training completes, restart the Chainlit app to load the new fine-tuned models.

For detailed training instructions, see [TRAINING_README.md](TRAINING_README.md).

## Troubleshooting

### Model Not Loading

If a model fails to load:
1. Check that the `.pth` file exists in the directory
2. Verify the file size matches expected size (not corrupted)
3. Check console output for specific error messages
4. Ensure you have enough RAM/VRAM for the model size

### Model Not Appearing in Dropdown

The app only shows models that successfully loaded. If a model is missing:
- Check the console output when starting the app
- Look for error messages about that specific model file
- Verify the model file name matches exactly what's in `app.py`

### Slow Response Times

If responses are slow:
- Use Small models instead of Medium for faster inference
- Ensure your system has adequate RAM (16GB+ recommended for Medium models)
- Close other memory-intensive applications
- On Apple Silicon, verify MPS is being used (check console output)

### Out of Memory Errors

If you get OOM errors:
- Use only Small models (require less memory)
- Reduce max_new_tokens in generation settings
- Free up system memory by closing other applications
