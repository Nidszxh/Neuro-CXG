# Deliverable 3: GPU Verification

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("CUDA computation successful.")
else:
    print("Running on CPU.")