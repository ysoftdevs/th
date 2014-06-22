
BINARIES = $(basename $(wildcard v*pu.c*))

all: $(BINARIES)

clean:
	rm -f $(BINARIES)

%-cpu: %-cpu.c
	gcc -std=c99 -pedantic -Wextra -O3 $< -lm -o $@

%-gpu: %-gpu.cu
	nvcc -gencode arch=compute_30,code=sm_30 -O3 $< -o $@
