#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>


#define MAX_PATH_LEN (32 * 1024)
#define MAX_KERNEL_RADIUS 16

// thread block size
#define TX 32
#define TY 32

struct kernel_params {
    float kernel[MAX_KERNEL_RADIUS + 1];
    int w;
    int h;
};

static void error(const char * message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(-1);
}

static void usage(const char * message, const char * app) {
    fprintf(stderr, "Usage: %s width height sigma file1 ... fileN\n", app);
    fprintf(stderr, "Example: %s 1920 1080 3 f1.gray f2.gray f3.gray\n", app);
    error(message);
}

static double timer_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec * 0.001;
}

static __device__ int saturate(int n, int max_value) {
    return max(0, min(n, max_value - 1));
}

static __device__ int get_pix(const uint8_t * src, int w, int h, int x, int y) {
    return (float)src[saturate(x, w) + saturate(y, h) * w];
}

static __global__ void convolution_vertical(kernel_params p, uint8_t * src, uint8_t * dest) {
    // coordinates of pixel processed by this thread
    const int x = threadIdx.x + blockIdx.x * TX;
    const int y = threadIdx.y + blockIdx.y * TY;
    
    // shared cache for processed pixels
    __shared__ float cache[TY + 2 * MAX_KERNEL_RADIUS][TX];
    
    // all threads populate shared cache
    for(int ny = 0; ny < 2; ny++) {
        cache[threadIdx.y + ny * TY][threadIdx.x]
                = get_pix(src, p.w, p.h, x, y - MAX_KERNEL_RADIUS + ny * TY);
    }
    
    // wait for all threads of block to finish their contribution to cache
    __syncthreads();
    
    // stop this thread if out of bounds
    if(x >= p.w || y >= p.h) {
        return;
    }
    
    // get weighted sum of neighbors
    float result = p.kernel[0] * cache[MAX_KERNEL_RADIUS + threadIdx.y][threadIdx.x];
    for(int k = 1; k <= MAX_KERNEL_RADIUS; k++) {
        result += p.kernel[k] * cache[MAX_KERNEL_RADIUS + threadIdx.y - k][threadIdx.x];
        result += p.kernel[k] * cache[MAX_KERNEL_RADIUS + threadIdx.y + k][threadIdx.x];
    }
    
    // save result
    dest[x + y * p.w] = saturate((int)result, 256);
}


static __global__ void convolution_horizontal(kernel_params p, uint8_t * src, uint8_t * dest) {
    // coordinates of pixel processed by this thread
    const int x = threadIdx.x + blockIdx.x * TX;
    const int y = threadIdx.y + blockIdx.y * TY;
    
    // shared cache for processed pixels
    __shared__ float cache[TY][TX + 2 * MAX_KERNEL_RADIUS];
    
    // all threads populate shared cache
    for(int nx = 0; nx < 2; nx++) {
        cache[threadIdx.y][threadIdx.x + nx * TX]
                = get_pix(src, p.w, p.h, x - MAX_KERNEL_RADIUS + nx * TX, y);
    }
    
    // wait for all threads of block to finish their contribution to cache
    __syncthreads();
    
    // stop this thread if out of bounds
    if(x >= p.w || y >= p.h) {
        return;
    }
    
    // get weighted sum of neighbors
    float result = p.kernel[0] * cache[threadIdx.y][MAX_KERNEL_RADIUS + threadIdx.x];
    for(int k = 1; k <= MAX_KERNEL_RADIUS; k++) {
        result += p.kernel[k] * cache[threadIdx.y][MAX_KERNEL_RADIUS + threadIdx.x + k];
        result += p.kernel[k] * cache[threadIdx.y][MAX_KERNEL_RADIUS + threadIdx.x - k];
    }
    
    // save result
    dest[x + y * p.w] = saturate((int)result, 256);
}


static float gaussian(float sigma, float x) {
    const float e = x / sigma;
    return exp(-0.5 * e * e);
}

int main(int argn, char ** argv) {
    kernel_params params;
    
    if(argn < 4) {
        usage("Wrong argument count", *argv);
    }
    
    // read width and height
    params.w = atoi(argv[1]);
    params.h = atoi(argv[2]);
    if(params.w < 1 || params.h < 1) {
        usage("Both width and height must be positive integers", *argv);
    }
    const int pix_count = params.w * params.h;
    
    // read sigma and prepare normalized kernel (sum = 1)
    const float sigma = atof(argv[3]);
    float kernel_sum = 0.0f;
    for(int k = 0; k <= MAX_KERNEL_RADIUS; k++) {
        kernel_sum += params.kernel[k] = gaussian(sigma, k);
    }
    kernel_sum = 2.0 * kernel_sum - params.kernel[0];
    for(int k = 0; k <= MAX_KERNEL_RADIUS; k++) {
        params.kernel[k] /= kernel_sum;
    }
    
    // dump the kernel
    printf("Convolution kernel:");
    for(int k = -MAX_KERNEL_RADIUS; k <= MAX_KERNEL_RADIUS; k++) {
        printf(" %f", params.kernel[k < 0 ? -k : k]);
    }
    printf("\n");
    
    // prepare buffers
    uint8_t * data_ptr;
    uint8_t * data_gpu_ptr[2];
    uint8_t * temp_gpu_ptr[2];
    cudaMalloc((void**)(data_gpu_ptr + 0), pix_count);
    cudaMalloc((void**)(temp_gpu_ptr + 0), pix_count);
    cudaMalloc((void**)(data_gpu_ptr + 1), pix_count);
    cudaMalloc((void**)(temp_gpu_ptr + 1), pix_count);
    cudaMallocHost((void**)&data_ptr, pix_count, cudaHostAllocDefault);
    
    // two CUDA streams for asynchronous kernel and data transfers
    cudaStream_t streams[2];
    cudaStreamCreate(streams + 0);
    cudaStreamCreate(streams + 1);
    
    // measure time of processing of all images
    const double begin = timer_ms();
    for(int i = 3; i <= argn; i++) {
        // index of I/O buffers in this iteration
        const int io_idx = i & 1;
        
        // index of computing resources in this iteration
        const int comp_idx = io_idx ^ 1;
        
        // start processing of image loaded in previous iteration 
        // (except of first and last iteration)
        if(i > 3 && i < argn) {
            // launch vertical and horizontal pass
            dim3 block(TX, TY);
            dim3 grid((params.w + TX - 1) / TX, (params.h + TY - 1) / TY);
            convolution_vertical<<<grid, block, 0, streams[comp_idx]>>>
                (params, data_gpu_ptr[comp_idx], temp_gpu_ptr[comp_idx]);
            convolution_horizontal<<<grid, block, 0, streams[comp_idx]>>>
                (params, temp_gpu_ptr[comp_idx], data_gpu_ptr[comp_idx]);
        }
        
        // processing now runs asynchronously on the GPU => save reauls 
        // from previous iteration (except of two first iterations)
        if(i > 4) {
            // copy data back from GPU
            cudaMemcpyAsync(data_ptr, data_gpu_ptr[io_idx], pix_count, 
                            cudaMemcpyDeviceToHost, streams[io_idx]);
            
            // compose output filename
            char out_path[MAX_PATH_LEN + 1];
            snprintf(out_path, MAX_PATH_LEN, "%s.out.gray", argv[i - 1]);
            
            // wait for the data to actually appear in the buffer
            cudaStreamSynchronize(streams[io_idx]);
            
            // write data to output file
            FILE * const out_file = fopen(out_path, "wb");
            if(NULL == out_file || 1 != fwrite(data_ptr, pix_count, 1, out_file)) {
                error(out_path);
            }
            fclose(out_file);
        }
        
        // load input for next iteration (except of two last iterations)
        if(i < (argn - 1)) {
            // read input file
            printf("Processing '%s'\n", argv[i + 1]);
            FILE * const src_file = fopen(argv[i + 1], "rb");
            if(NULL == src_file || 1 != fread(data_ptr, pix_count, 1, src_file)) {
                error(argv[i + 1]);
            }
            fclose(src_file);
            
            // copy data to GPU memory
            cudaMemcpyAsync(data_gpu_ptr[io_idx], data_ptr, pix_count, 
                            cudaMemcpyHostToDevice, streams[io_idx]);
            
            // make sure that the buffer is ready for next iteration
            cudaStreamSynchronize(streams[io_idx]);
        }
    }
    const double end = timer_ms();
    
    // print total time
    printf("time: %f ms, %d images => %f ms/image\n",
           end - begin, argn - 4, (end - begin) / (argn - 4));
    
    // cleanup
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFree(data_gpu_ptr[0]);
    cudaFree(temp_gpu_ptr[0]);
    cudaFree(data_gpu_ptr[1]);
    cudaFree(temp_gpu_ptr[1]);
    cudaFreeHost(data_ptr);
    return 0;
}

