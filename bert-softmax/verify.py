import torch
import numpy as np

size = 4
# Load matrices from files
h_in = np.loadtxt('matrix_in.txt').reshape((size, size))

# Convert numpy matrices to PyTorch tensors
in_mat = torch.from_numpy(h_in).float()

# Perform softmax using PyTorch
result = torch.nn.functional.softmax(in_mat, dim=1)

# Load result matrix from file
result_matrix = np.loadtxt('result_matrix.txt').reshape((size, size))

# Compare results
if torch.allclose(result, torch.from_numpy(result_matrix).float()):
    print("Softmax results match with the results saved in the file.")
    for i in range(size):
        for j in range(size):
            expected = result_matrix[i][j]
            real = result[i][j]
            if expected != real:
                print(f"At index ({i},{j}): Expected - {expected}, Real - {real}")
else:
    print("Softmax results do not match with the results saved in the file.")
    for i in range(size):
        for j in range(size):
            expected = result_matrix[i][j]
            real = result[i][j]
            if expected != real:
                print(f"At index ({i},{j}): Expected - {expected}, Real - {real}")
