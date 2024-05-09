__kernel void layer_norm(__global float* input, 
                         __global float* output,
                         int width, 
                         int height) {

    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < width && col < height) {
        float sum = 0.0f;
        float mean = 0.0f;
        float variance = 0.0f;
        float epsilon = 1e-5f;

        // Calculate mean
        for (int i = 0; i < height; i++) {
            sum += input[row * height + i];
        }
        mean = sum / height;

        // Calculate variance
        sum = 0.0f;
        for (int i = 0; i < height; i++) {
            sum += pow(input[row * height + i] - mean, 2);
        }
        variance = sum / height;

        // Normalize input
        output[row * height + col] = (input[row * height + col] - mean) / sqrt(variance + epsilon);
    }
}