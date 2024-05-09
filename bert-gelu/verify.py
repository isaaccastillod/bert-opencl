import torch
import numpy as np

size = 1024
# Load matrices from files
h_a = np.loadtxt('matrix_a.txt').reshape((size, size))

# Convert numpy matrix to PyTorch tensor
a = torch.from_numpy(h_a).float()

# Perform gelu over matrix
result = torch.nn.functional.gelu(a)

# Load result matrix from file
result_matrix = np.loadtxt('result_matrix.txt').reshape((size, size))

# Compare results
if torch.allclose(result, torch.from_numpy(result_matrix).float()):
    print("GELU operation results match with the results saved in the file.")
else:
    print("GELU operation results do not match with the results saved in the file.")
    for i in range(size):
        for j in range(size):
            expected = result_matrix[i][j]
            real = result[i][j]
            if expected != real:
                print(f"At index ({i},{j}): Expected - {expected}, Real - {real}")
