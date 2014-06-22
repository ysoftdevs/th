#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>


#define MAX_PATH_LEN (32 * 1024)
#define MAX_KERNEL_RADIUS 16

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

static int saturate(int n, int max) {
    return n < 0 ? 0 : (n < max ? n : max - 1);
}

static int get_pix(const uint8_t * src, int w, int h, int x, int y) {
    return src[saturate(x, w) + saturate(y, h) * w];
}

static float gaussian(float sigma, float x) {
    const float e = x / sigma;
    return exp(-0.5 * e * e);
}

int main(int argn, char ** argv) {
    if(argn < 4) {
        usage("Wrong argument count", *argv);
    }
    
    // read width and height
    const int w = atoi(argv[1]);
    const int h = atoi(argv[2]);
    if(w < 1 || h < 1) {
        usage("Both width and height must be positive integers", *argv);
    }
    const int pix_count = w * h;
    
    // read sigma and prepare normalized kernel (sum = 1)
    const float sigma = atof(argv[3]);
    float kernel[MAX_KERNEL_RADIUS + 1];
    float kernel_sum = 0.0f;
    for(int k = 0; k <= MAX_KERNEL_RADIUS; k++) {
        kernel_sum += kernel[k] = gaussian(sigma, k);
    }
    kernel_sum = 2.0 * kernel_sum - kernel[0];
    for(int k = 0; k <= MAX_KERNEL_RADIUS; k++) {
        kernel[k] /= kernel_sum;
    }
    
    // dump the kernel
    printf("Convolution kernel:");
    for(int k = -MAX_KERNEL_RADIUS; k <= MAX_KERNEL_RADIUS; k++) {
        printf(" %f", kernel[k < 0 ? -k : k]);
    }
    printf("\n");
    
    // prepare buffers
    uint8_t * const data_ptr = (uint8_t*)malloc(pix_count);
    uint8_t * const temp_ptr = (uint8_t*)malloc(pix_count);
    
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
        
        // vertical pass: for each pixel
        uint8_t * out_pix_ptr = temp_ptr;
        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                // sum up all weighted neighbors and the pixel itself
                float result = kernel[0] * get_pix(data_ptr, w, h, x, y);
                for(int k = 1; k <= MAX_KERNEL_RADIUS; k++) {
                    result += kernel[k] * (get_pix(data_ptr, w, h, x, y + k)
                                         + get_pix(data_ptr, w, h, x, y - k));
                }
                *(out_pix_ptr++) = saturate((int)result, 256);
            }
        }
        
        // horizontal pass: for each pixel
        out_pix_ptr = data_ptr;
        for(int y = 0; y < h; y++) {
            for(int x = 0; x < w; x++) {
                // sum up all weighted neighbors and the pixel itself
                float result = kernel[0] * get_pix(temp_ptr, w, h, x, y);
                for(int k = 1; k <= MAX_KERNEL_RADIUS; k++) {
                    result += kernel[k] * (get_pix(temp_ptr, w, h, x + k, y)
                                         + get_pix(temp_ptr, w, h, x - k, y));
                }
                *(out_pix_ptr++) = saturate((int)result, 256);
            }
        }
        
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
    free(temp_ptr);
    free(data_ptr);
    return 0;
}

