#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


__kernel void sgemm_naive(int M, int N, int K, float alpha, __global const float *A,
                            __global const float *B, float beta, __global float *C) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  // `if` condition is necessary for when M or N aren't multiples of the work-group size.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    
  }
}


// __kernel void sgemm_shared_mem_block(int M, int N, int K,
//                                      float alpha, __global const float* A, __global const float* B,
//                                      float beta, __global float* C) {
//   // the output block that we want to compute in this workgroup
//   const uint cRow = get_group_id(0);
//   const uint cCol = get_group_id(1);

//   // allocate buffer for current block in local memory
//   // local mem is shared between all work-items in a workgroup
//   __local float As[BLOCKSIZE * BLOCKSIZE];
//   __local float Bs[BLOCKSIZE * BLOCKSIZE];

//   // the inner row & col that we're accessing in this work-item
//   const uint threadCol = get_local_id(0) % BLOCKSIZE;
//   const uint threadRow = get_local_id(0) / BLOCKSIZE;

//   // advance pointers to the starting positions
//   A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
//   B += cCol * BLOCKSIZE;                          // row=0, col=cCol
//   C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

//   float tmp = 0.0;
//   for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
//     // Have each work-item load one of the elements in A & B
//     // Make the threadCol (=get_local_id(0)) the consecutive index
//     // to allow global memory access coalescing
//     As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
//     Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

//     // work-items in this workgroup need to sync to ensure cache is fully populated
//     barrier(CLK_LOCAL_MEM_FENCE);
//     A += BLOCKSIZE;
//     B += BLOCKSIZE * N;

//     // execute the dot product on the currently cached block
//     for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
//       tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
//     }
//     // need to sync again at the end to avoid faster work-items fetching the next block before slower work-items are done
//     barrier(CLK_LOCAL_MEM_FENCE);
//   }
//   C[threadRow * N + threadCol] =
//       alpha * tmp + beta * C[threadRow * N + threadCol];
// }

__kernel void transpose(__global float* input, __global float* output, const int width, const int height) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    output[i * height + j] = input[j * width + i];
}

__kernel void layerNormalization(__global float* input,
                                 __global float* output,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
    
    int gid = get_global_id(0);
    
    // Calculate mean and variance for each channel
    float mean = 0.0f;
    float variance = 1.0f;
    
    for (int i = 0; i < height * width; i++) {
        mean += input[gid * height * width + i];
    }
    mean /= height * width;
    
    for (int i = 0; i < height * width; i++) {
        variance += pow(input[gid * height * width + i] - mean, 2);
    }
    variance /= height * width;
    
    // Apply normalization
    for (int i = 0; i < height * width; i++) {
        output[gid * height * width + i] = (input[gid * height * width + i] - mean) / sqrt(variance + 0.0001f);
    }
}


