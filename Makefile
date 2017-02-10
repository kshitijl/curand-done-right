NVCC= nvcc --expt-extended-lambda -std=c++11
DIR=src/curand-done-right
HEADER=$(DIR)/curanddr.hxx

all: objdir bin/thrust-example bin/mgpu-example

objdir: bin

bin:
	mkdir -p bin

bin/thrust-example: examples/thrust.cu $(HEADER)
	$(NVCC) examples/thrust.cu -Isrc -o bin/thrust-example

bin/mgpu-example: examples/mgpu.cu $(HEADER)
	$(NVCC) examples/mgpu.cu -Isrc -Iexamples/moderngpu/src -o bin/mgpu-example
