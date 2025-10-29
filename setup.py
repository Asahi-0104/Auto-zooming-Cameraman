import sys
import subprocess

def install_with_progress(package_cmd):
    print("-" * 50)
    result = subprocess.run(package_cmd, shell=True)
    print("-" * 50)
    if result.returncode != 0:
        print("Installation failed!")
        sys.exit(1)
    print("Installation successful！\n")

def main():
    print("Installing YOLO...")

    # Install PyTorch with CUDA
    torch_cmd = (
        f'"{sys.executable}" -m pip install torch torchvision torchaudio '
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    install_with_progress(torch_cmd)

    # Install YOLO
    yolo_cmd = f'"{sys.executable}" -m pip install ultralytics'
    install_with_progress(yolo_cmd)

    try:
        import torch
        from ultralytics import __version__ as yolo_ver
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
        print(f"YOLO {yolo_ver} installed!")
        print(f"PyTorch: {torch.__version__}")
        print(f"Device:  {device} ({gpu})")
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()