import torch
import onnxruntime as ort

print("=== GPU ACCELERATION CHECK ===")
print("PyTorch Version:", torch.__version__)
print("PyTorch CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
    print("CUDA Capability:", torch.cuda.get_device_capability(0))
    
    # Quick tensor test on GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print("GPU Matrix Multiplication Test: SUCCESS! (Tensor shape:", y.shape, ")")

print("\nONNX Runtime Version:", ort.__version__)
print("ONNX Available Providers:", ort.get_available_providers())
print("==============================")
