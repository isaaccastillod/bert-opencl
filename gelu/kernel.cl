__kernel void gelu(__global float* const input, __global float* output, const int rows, const int cols) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < rows && col < cols) {
        // Calculate the index of the element in the input matrix
        int index = row * cols + col;

        // Apply the GELU function to each element in the matrix
        float x = input[index];
        printf("Element before gelu %f\n", x);
        float y = 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
        printf("Element after gelu %f\n", y);

        // Store the result back in the input matrix
        output[index] = y;
    }
}