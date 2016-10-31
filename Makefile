NVCC= nvcc --expt-extended-lambda -std=c++11
DIR=src/curand-done-right
HEADER=$(DIR)/curanddr.hxx

all: objdir bin/thrust-example

objdir: bin

bin:
	mkdir -p bin

bin/thrust-example: examples/thrust.cu $(HEADER)
	$(NVCC) examples/thrust.cu -I$(DIR) -o bin/thrust-example

