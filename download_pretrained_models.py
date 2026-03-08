"""
Download GPT-2 Small (124M) and Medium (355M) models from HuggingFace
"""
import os
import urllib.request

def download_model(file_name, description):
    """
    Download a GPT-2 model from HuggingFace.
    
    Args:
        file_name: Name of the model file (e.g., "gpt2-small-124M.pth")
        description: Human-readable description (e.g., "GPT-2 Small (124M)")
    """
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"
    
    if os.path.exists(file_name):
        file_size = os.path.getsize(file_name) / (1024**3)  # Size in GB
        print(f"✓ {file_name} already exists ({file_size:.2f} GB)")
        return True
    
    print(f"\nDownloading {description}...")
    print(f"File: {file_name}")
    print(f"URL: {url}")
    
    # Estimate size based on model
    if "124M" in file_name:
        size_str = "~650 MB"
    elif "355M" in file_name:
        size_str = "~1.4 GB"
    else:
        size_str = "large"
    
    print(f"Expected size: {size_str}")
    print("This may take a while...")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / (1024**2)
            total_mb = total_size / (1024**2)
            print(f"\rDownloading: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="")
        
        urllib.request.urlretrieve(url, file_name, progress_hook)
        print(f"\n✓ Downloaded {file_name}")
        
        # Verify file size
        file_size = os.path.getsize(file_name) / (1024**3)
        print(f"File size: {file_size:.2f} GB")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {file_name}: {e}")
        if os.path.exists(file_name):
            os.remove(file_name)
        return False

if __name__ == "__main__":
    print("="*80)
    print("GPT-2 Model Download (Small 124M + Medium 355M)")
    print("="*80)
    
    models = [
        ("gpt2-small-124M.pth", "GPT-2 Small (124M)"),
        ("gpt2-medium-355M.pth", "GPT-2 Medium (355M)")
    ]
    
    success_count = 0
    for file_name, description in models:
        if download_model(file_name, description):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"✓ {success_count}/{len(models)} model(s) ready")
    
    if success_count == len(models):
        print("✓ All models downloaded! You can now run:")
        print("  - bin/python train_san_francisco.py  (to fine-tune both models)")
        print("  - bin/chainlit run app.py --port 8000  (to use the models)")
    elif success_count > 0:
        print("⚠ Some models are available, but not all downloads succeeded")
    else:
        print("❌ No models were downloaded successfully")
    
    print("="*80)
