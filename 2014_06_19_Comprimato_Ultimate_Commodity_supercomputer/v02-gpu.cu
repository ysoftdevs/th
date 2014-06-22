#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>


#define MAX_PATH_LEN (32 * 1024)
#define MAX_KERNEL_RADIUS 16

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

template <int DX, int DY>
static __global__ void convolution(kernel_params p, uint8_t * src, uint8_t * dest) {
    // coordinates of processed pixel
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // stop if out of bounds
    if(x >= p.w || y >= p.h) {
        return;
    }
    
    // get weighted sum of neighbors
    float result = p.kernel[0] * get_pix(src, p.w, p.h, x, y);
    for(int k = 1; k <= MAX_KERNEL_RADIUS; k++) {
        result += p.kernel[k] * (get_pix(src, p.w, p.h, x + k * DX, y + k * DY)
                               + get_pix(src, p.w, p.h, x - k * DX, y - k * DY));
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
    uint8_t * const data_ptr = (uint8_t*)malloc(pix_count);
    uint8_t * data_gpu_ptr;
    uint8_t * temp_gpu_ptr;
    cudaMalloc((void**)&data_gpu_ptr, pix_count);
    cudaMalloc((void**)&temp_gpu_ptr, pix_count);
    
    // measure time of processing of all images
    const double begin = timer_ms();
    for(int i = 4; i < argn; i++) {
        // read input data
        printf("Processing '%s'\n", argv[i]);
        FILE * const src_file = fopen(argv[i], "rb");
        if(NULL == src_file || 1 != fread(data_ptr, pix_count, 1, src_file)) {
            error(argv[i]);
        }
        fclose(src_file);
        
        // copy data to GPU memory
        cudaMemcpy(data_gpu_ptr, data_ptr, pix_count, cudaMemcpyHostToDevice);
        
        // launch vertical and horizontal pass
        dim3 block(32, 32);
        dim3 grid((params.w + block.x - 1) / block.x,
                  (params.h + block.y - 1) / block.y);
        convolution<0, 1><<<grid, block>>>(params, data_gpu_ptr, temp_gpu_ptr);
        convolution<1, 0><<<grid, block>>>(params, temp_gpu_ptr, data_gpu_ptr);
        
        // copy data back from GPU
        cudaMemcpy(data_ptr, data_gpu_ptr, pix_count, cudaMemcpyDeviceToHost);
        
        // compose output filename
        char out_path[MAX_PATH_LEN + 1];
        snprintf(out_path, MAX_PATH_LEN, "%s.out.gray", argv[i]);
        
        // write data to output file
        FILE * const out_file = fopen(out_path, "wb");
        if(NULL == out_file || 1 != fwrite(data_ptr, pix_count, 1, out_file)) {
            error(out_path);
        }
        fclose(out_file);
    }
    const double end = timer_ms();
    
    // print total time
    printf("time: %f ms, %d images => %f ms/image\n",
           end - begin, argn - 4, (end - begin) / (argn - 4));
    
    // cleanup
    free(data_ptr);
    cudaFree(data_gpu_ptr);
    cudaFree(temp_gpu_ptr);
    return 0;
}

