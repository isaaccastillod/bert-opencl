__kernel void softmax(__global const float *input, __global float *output, int const rows, int const cols) {
    int global_id = get_global_id(0);

    if (global_id < rows) {
        float max_val = input[global_id * cols];
        float sum = 0.0f;

        // Find the maximum value in the row
        for (int j = 1; j < cols; j++) {
            if (input[global_id * cols + j] > max_val) {
                max_val = input[global_id * cols + j];
            }
        }

        // Compute the softmax function for the row
        for (int j = 0; j < cols; j++) {
            output[global_id * cols + j] = exp(input[global_id * cols + j] - max_val);
            sum += output[global_id * cols + j];
        }

        // Normalize the output row
        for (int j = 0; j < cols; j++) {
            output[global_id * cols + j] /= sum;
        }
    }
}