import torch


def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs tensor.
    
    Args:
        text: Input text string
        tokenizer: Tokenizer instance (e.g., tiktoken)
        
    Returns:
        Tensor of token IDs with batch dimension
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs tensor back to text.
    
    Args:
        token_ids: Tensor of token IDs
        tokenizer: Tokenizer instance (e.g., tiktoken)
        
    Returns:
        Decoded text string
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generate text using the model with temperature and top-k sampling.
    
    Args:
        model: The GPT model
        idx: Starting token indices (batch, n_tokens)
        max_new_tokens: Maximum number of new tokens to generate
        context_size: Maximum context length the model can handle
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_k: If set, only sample from top k tokens
        eos_id: End of sequence token id to stop generation early
        
    Returns:
        Tensor of generated token IDs
    """
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]
        
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        logits = logits[:, -1, :]
        
        # Filter with top-k sampling
        if top_k is not None:
            # Keep only top k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy sampling
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Check for end of sequence
        if idx_next == eos_id:
            break
        
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
