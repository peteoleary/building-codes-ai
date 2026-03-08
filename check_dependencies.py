#!/usr/bin/env python
"""
Verify that all required dependencies are installed and working correctly.
Run this script before using app.py or train_san_francisco.py
"""
import sys

def check_python_version():
    """Check if Python version is 3.13+"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 13):
        print("  ❌ Python 3.13+ required")
        return False
    else:
        print("  ✓ Python version OK")
        return True


def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name} {version}")
        return True
    except ImportError as e:
        print(f"  ❌ {package_name} not installed: {e}")
        return False


def check_pytorch_device():
    """Check PyTorch device availability"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device = "CUDA"
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU available: {device_name}")
        elif torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                print(f"  ✓ MPS available (PyTorch {major}.{minor})")
            else:
                print(f"  ⚠ MPS available but PyTorch {major}.{minor} < 2.9 (may be unstable)")
        else:
            device = "CPU"
            print(f"  ✓ CPU only (no GPU acceleration)")
        
        return True
    except Exception as e:
        print(f"  ❌ Error checking PyTorch device: {e}")
        return False


def check_local_modules():
    """Check if local project modules are accessible"""
    modules = ['gpt_model', 'gpt_utils', 'gpt_dataset']
    all_ok = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}.py")
        except ImportError as e:
            print(f"  ❌ {module}.py not found or has errors: {e}")
            all_ok = False
    
    return all_ok


def check_data_files():
    """Check if required data files exist"""
    import os
    
    files_to_check = {
        'san_francisco-ca-1.txt': 'Training data (required for training)',
        'gpt2-small-124M.pth': 'Small pretrained model (optional)',
        'gpt2-medium-355M.pth': 'Medium pretrained model (optional)',
    }
    
    found_any_model = False
    
    for filename, description in files_to_check.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024**3)
            print(f"  ✓ {filename} ({size:.2f} GB) - {description}")
            if 'model' in description.lower():
                found_any_model = True
        else:
            if 'required' in description.lower():
                print(f"  ❌ {filename} - {description}")
            else:
                print(f"  ⚠ {filename} not found - {description}")
    
    return found_any_model


def main():
    print("="*80)
    print("DEPENDENCY CHECK FOR GPT-2 TRAINING & CHAINLIT APP")
    print("="*80)
    
    all_checks = []
    
    # Python version
    print("\n1. Python Version:")
    all_checks.append(check_python_version())
    
    # Core dependencies
    print("\n2. Core Dependencies:")
    all_checks.append(check_import('torch', 'PyTorch'))
    all_checks.append(check_import('tiktoken'))
    
    # PyTorch device
    print("\n3. PyTorch Device:")
    all_checks.append(check_pytorch_device())
    
    # Chainlit (optional for training)
    print("\n4. Chainlit (for app.py):")
    chainlit_ok = check_import('chainlit', 'Chainlit')
    if not chainlit_ok:
        print("     Note: Chainlit is only needed for app.py, not for training")
    
    # Local modules
    print("\n5. Local Project Modules:")
    all_checks.append(check_local_modules())
    
    # Data files
    print("\n6. Data Files:")
    has_model = check_data_files()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all(all_checks):
        print("✓ All required dependencies are installed!")
        
        if not has_model:
            print("\n⚠ WARNING: No model files found")
            print("   Download models with:")
            print("     bin/python download_gpt2_medium.py")
        
        if not chainlit_ok:
            print("\n⚠ WARNING: Chainlit not installed")
            print("   Install with: pip install chainlit")
            print("   (Only needed for app.py)")
        
        print("\nYou can now:")
        print("  • Run training: bin/python train_san_francisco.py")
        if chainlit_ok:
            print("  • Run app: bin/chainlit run app.py --port 8000")
    else:
        print("❌ Some dependencies are missing!")
        print("\nTo install dependencies, run:")
        print("  pip install -r requirements.txt")
        print("\nFor PyTorch with specific hardware:")
        print("  • Apple Silicon: pip install torch>=2.9.0")
        print("  • NVIDIA GPU: Visit https://pytorch.org for CUDA installation")
    
    print("="*80)


if __name__ == "__main__":
    main()
