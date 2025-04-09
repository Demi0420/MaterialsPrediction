import torch

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")