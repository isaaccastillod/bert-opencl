import torch
import numpy as np

size = 1024
# Load matrices from files
h_a = np.loadtxt('matrix_a.txt').reshape((size, size))
h_b = np.loadtxt('matrix_b.txt').reshape((size, size))

# Convert numpy matrices to PyTorch tensors
a = torch.from_numpy(h_a).float()
b = torch.from_numpy(h_b).float()

# Perform matrix multiplication using PyTorch
result = torch.matmul(a, b)

# Load result matrix from file
result_matrix = np.loadtxt('result_matrix.txt').reshape((size, size))

# Compare results
if torch.allclose(result, torch.from_numpy(result_matrix).float()):
    print("Matrix multiplication results match with the results saved in the file.")
else:
    print("Matrix multiplication results do not match with the results saved in the file.")
