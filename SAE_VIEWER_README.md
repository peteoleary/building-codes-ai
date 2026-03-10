# SAE Feature Viewer

Interactive web application to explore and compare interpretable features learned by sparse autoencoders on pretrained vs fine-tuned GPT-2 models.

## Quick Start

```bash
# Launch the viewer
bin/streamlit run sae_viewer.py

# Or with custom port
bin/streamlit run sae_viewer.py --server.port 8501
```

The viewer will open in your browser at `http://localhost:8501`

## Features

### 1. 🔤 Text Analysis
- Enter any text to see which SAE features activate
- Compare feature activations between pretrained and fine-tuned models side-by-side
- See tokenization and top-10 most activated features
- Identify "novel" features that are much more active in the fine-tuned model
- Identify "lost" features that were active in pretrained but suppressed by fine-tuning

**Example Use:**
- Enter building code text: "All buildings shall comply with fire safety requirements under Section 101."
- See which features activate more strongly in the fine-tuned model
- These are the features that learned building code concepts!

### 2. ⚖️ Model Comparison
- Run batch comparison on multiple test texts
- See how general English vs building code text activates different features
- Quickly identify which features are building-code-specific

**Test Texts Included:**
- Building code sentences (fire safety, egress, sections)
- General English sentences (for baseline comparison)

### 3. 🔬 Feature Explorer
- Examine individual features in detail
- See decoder weights and statistics for any feature
- Test feature activation on custom text
- Understand what each feature represents

**How to Use:**
1. Select pretrained or fine-tuned model
2. Enter a feature ID (0 to 8191)
3. View decoder weight statistics
4. Test with custom text to see activation values

### 4. 🏗️ Building Code Concepts
- Pre-configured building code concepts:
  - **Fire Safety**: fire safety, sprinklers, fire exits, etc.
  - **Structural Requirements**: load-bearing, seismic, foundation, etc.
  - **Occupancy**: occupancy classification, load, types, etc.
  - **Egress**: means of egress, exit access, width requirements, etc.
  - **Section References**: Section 101, Section 102, Chapter 10, etc.

- Find which features most strongly represent each concept
- Compare concept representations between pretrained and fine-tuned models
- Discover if fine-tuning created dedicated features for building code concepts

## What to Look For

### Novel Features (Fine-tuned Only)
Features that activate strongly in the fine-tuned model but weakly in pretrained indicate building-code-specific concepts learned during training:
- "Section" + number patterns
- "shall" + requirement language
- Fire safety terminology
- Egress and exit concepts
- Occupancy classifications

### Preserved Features (Both Models)
Features that activate similarly in both models represent general language patterns:
- Common words and phrases
- Grammatical structures
- General semantic concepts

### Lost Features (Pretrained Only)
Features that were active in pretrained but suppressed in fine-tuned show what was de-emphasized:
- Casual language patterns
- Conversational structures
- Non-technical vocabulary

## Configuration

**Sidebar Options:**
- **Model Size**: Small (124M) or Medium (355M)
- **Layer**: Which transformer layer to analyze (default: 6, middle layer)

**Model Requirements:**
- Pretrained SAE: `sae_small_pretrained_layer6.pth`
- Fine-tuned SAE: `sae_small_finetuned_layer6.pth`
- Pretrained GPT-2: `gpt2-small-124M.pth`
- Fine-tuned GPT-2: `gpt2-san-francisco-finetuned.pth`

## Example Workflow

### Discover Building Code Features

1. **Start with Text Analysis Tab**
   - Enter: "All buildings shall comply with Section 101."
   - Note which features have high activation in fine-tuned but low in pretrained
   - Write down the feature IDs

2. **Explore Those Features**
   - Go to Feature Explorer tab
   - Select fine-tuned model
   - Enter one of the feature IDs you found
   - Test with variations: "Section 102", "Section 103", "Chapter 10"
   - See if the feature consistently activates for section references

3. **Check Concept Groupings**
   - Go to Building Code Concepts tab
   - Select "Section References" concept
   - See which features represent this concept
   - Compare pretrained vs fine-tuned

4. **Compare Models**
   - Go to Model Comparison tab
   - Run comparison on the test texts
   - See how building code text differs from general English

### Find Specific Concept Features

**Looking for "fire safety" features:**
1. Building Code Concepts → "Fire Safety"
2. Analyze → Note top 10 features
3. Feature Explorer → Test each feature with fire safety texts
4. Text Analysis → Enter custom fire safety text to verify

**Looking for "shall" (legal requirement) features:**
1. Text Analysis → Enter: "The building shall provide adequate egress."
2. Note top activated features
3. Feature Explorer → Test those features with other "shall" sentences
4. Compare with "must" or "should" to see specificity

## Tips

- **Use the sidebar** to switch between layers (0-11) to see how features differ across model depth
- **Try edge cases**: Mix building code terms with general language to see feature specificity
- **Compare similar concepts**: "fire safety" vs "fire alarm" vs "sprinkler" to see granularity
- **Look for compositionality**: Do features activate for "Section 101" but not "Section" alone?

## Troubleshooting

**"Could not load SAE model" error:**
- Make sure you've trained both SAE models:
  ```bash
  bin/python train_sparse_autoencoder.py --model small --checkpoint pretrained
  bin/python train_sparse_autoencoder.py --model small --checkpoint finetuned
  ```

**"Could not load GPT model" error:**
- Ensure you have the pretrained and fine-tuned GPT-2 models
- Check file names match the expected names

**Slow performance:**
- Streamlit caches models, first load may be slow
- Subsequent analyses are much faster
- Consider using smaller text samples for faster iteration

## Architecture

**SAE Structure:**
- Input: 768D (Small) or 1024D (Medium) activation vectors from GPT-2 layer
- Hidden: 8192 sparse features (ReLU activation)
- Output: Reconstruction to original activation space
- Loss: MSE reconstruction + L1 sparsity penalty

**Feature Interpretation:**
- Each of 8192 features should activate for specific concepts
- Sparsity ensures only ~5-10% of features activate for any given token
- Decoder weights show what each feature represents in the model's internal space

## Next Steps

After exploring features in the viewer:

1. **Save interesting features**: Document which feature IDs represent which concepts
2. **Feature clamping experiments**: Modify [gpt_model.py](gpt_model.py) to clamp specific features
3. **Generate with clamped features**: See how forcing features on/off changes generated text
4. **Train on other layers**: Try layers 0-11 to see which layer learns the best features
5. **Scale up**: Train on Medium model for potentially more refined features

## Related Files

- [train_sparse_autoencoder.py](train_sparse_autoencoder.py) - Train SAE on model activations
- [analyze_sae_features.py](analyze_sae_features.py) - Find top activating examples for features
- [compare_models_features.py](compare_models_features.py) - Statistical comparison between models
- [SPARSE_AUTOENCODER_README.md](SPARSE_AUTOENCODER_README.md) - Full SAE documentation
