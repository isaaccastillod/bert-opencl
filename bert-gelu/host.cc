#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h> 
#include <string.h>
#include <chrono>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>

#define KERNEL_FILE "kernel.cl"
#define KERNEL_NAME "gelu"

#define MAX_SOURCE_SIZE (0x100000)

#define FLOAT_ULP 6

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
	 cleanup();			                                                     \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
	   cleanup();			                                                   \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

static void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time;
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  printf("Generating matrix values \n");
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

static void save_matrix_to_file(float* matrix, int size, const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
  std::cerr << "Error opening file for writing: " << filename << std::endl;
  return;
  }
  for (int i = 0; i < size * size; i++) {
    if (i % size == 0 && i != 0) {
        file << "\n";
    }
    file << matrix[i] << " ";
  }

  file.close();
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem in_memobj = NULL;
cl_mem out_memobj = NULL; 
float *h_in = NULL;
float *h_out = NULL;
uint8_t *kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (in_memobj) clReleaseMemObject(in_memobj);
  if (out_memobj) clReleaseMemObject(out_memobj);     
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  
  if (kernel_bin) free(kernel_bin);
  if (h_in) free(h_in);
  if (h_out) free(h_out);
}


static void parse_args(int argc, char **argv) {

}

int main(int argc, char ** argv) {
    cl_platform_id platform_id;
    size_t kernel_size;

    // Get platform and device information
    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

    printf("Create context\n");
    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
    int size = 16;

    printf("Allocate device buffers\n");
    size_t nbytes = size * size * sizeof(float);
    in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
    out_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

    printf("Create program from kernel source\n");
    if (0 != read_kernel_file(KERNEL_FILE, &kernel_bin, &kernel_size)) {
      return -1;
    }
    program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));  
    
    // Build program
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));
    
    // Create kernel
    kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));
    int cols = size;
    int rows = size;
    // Set kernel arguments
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_memobj));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&out_memobj));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void *)&rows));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void *)&cols));	


    // Generate input values
    h_in = (float *)malloc(sizeof(float) * size * size);
    h_out = (float *)malloc(sizeof(float) * size * size);
    printf("Generating INPUT matrix\n");
    randomize_matrix(h_in, size * size);
    
    // Write matrices to file
    save_matrix_to_file(h_in, size, "matrix_in.txt");

    size_t global_offset[2] = {0, 0};
    size_t global_work_size[2] = {size, size};
    size_t local_work_size[2] = {size, size};    

    // Creating command queue
    printf("Creating command queue \n");
    commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));  

      printf("Upload source buffers\n");
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, nbytes, h_in, 0, NULL, NULL));

   
    printf("Execute the kernel\n");  
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, global_offset, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

    printf("Download destination buffer\n");
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, nbytes, h_out, 0, NULL, NULL));

    // Write results to file
    save_matrix_to_file(h_out, size, "result_matrix.txt");


    printf("Print C values \n");
    for (int i = 0; i < size*size; i++) {
      printf("%f\n", h_out[i]);
    }
    cleanup();
}