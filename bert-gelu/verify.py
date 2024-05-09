import torch
import numpy as np

size = 16
# Load matrices from files
h_in = np.loadtxt('matrix_in.txt').reshape((size, size))

# Convert numpy matrix to PyTorch tensor
input_matrix = torch.from_numpy(h_in).float()

# Perform gelu over matrix
result = torch.nn.functional.gelu(input_matrix)

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
