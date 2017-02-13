NVCC=nvcc --expt-extended-lambda -std=c++11 -Wno-deprecated-gpu-targets
DIR=src/curand-done-right
HEADER=$(DIR)/curanddr.hxx

all: objdir bin/thrust-example bin/mgpu-example bin/basic-pi-example bin/mgpu-pi-example bin/miller-rabin

objdir: bin

bin:
	mkdir -p bin

bin/thrust-example: examples/thrust.cu $(HEADER)
	$(NVCC) examples/thrust.cu -Isrc -o bin/thrust-example

bin/mgpu-example: examples/mgpu.cu $(HEADER)
	$(NVCC) examples/mgpu.cu -Isrc -Iexamples/moderngpu/src -o bin/mgpu-example

bin/basic-pi-example: examples/basic-pi.cu $(HEADER)
	$(NVCC) examples/basic-pi.cu -Isrc -o bin/basic-pi-example

bin/mgpu-pi-example: examples/mgpu-pi.cu $(HEADER)
	$(NVCC) examples/mgpu-pi.cu -Isrc -Iexamples/moderngpu/src -o bin/mgpu-pi-example

bin/miller-rabin: examples/miller-rabin.cu $(HEADER)
	$(NVCC) examples/miller-rabin.cu -Isrc -Iexamples/moderngpu/src -o bin/miller-rabin
