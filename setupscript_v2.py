import os
import subprocess
import sys
import urllib.request
import platform

# CUDA installer information
CUDA_PATH_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
CUDA_PATH_LIB = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib"

def set_cuda_paths():
    """Sets CUDA paths in the environment."""
    os.environ["PATH"] += os.pathsep + CUDA_PATH_BIN
    os.environ["PATH"] += os.pathsep + CUDA_PATH_LIB
    print("CUDA paths set.")

def install_requirements():
    """Installs dependencies in the current Python environment with the custom PyTorch URL for CUDA 11.8."""
    # Install PyTorch with CUDA 11.8 support
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])

    # Install remaining dependencies
    other_requirements = [
        "faster-whisper",
        "sounddevice",
        "numpy",
        "keyboard"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + other_requirements)
    print("All dependencies installed.")

def main():
    set_cuda_paths()
    install_requirements()
    print("Setup completed in the current environment.")

if __name__ == "__main__":
    main()
