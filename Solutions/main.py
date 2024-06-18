import torch
# Check if GPU is available, use CPU if not
if torch.backends.mps.is_available():
    device = 'mps'
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is Apple MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is Apple MPS available? {torch.backends.mps.is_available()}")
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Using device: {device}")