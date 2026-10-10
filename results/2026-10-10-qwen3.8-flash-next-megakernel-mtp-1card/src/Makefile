LLAMA ?= ../llama.cpp
BUILD ?= $(LLAMA)/build-sm80
ARCH  ?= sm_80
all: fnx_mk fnx_spec
fnx_mk: fnx_mk.cu fnx_tables.h
	nvcc -O3 -std=c++17 -arch=$(ARCH) -lineinfo -I$(LLAMA)/include -I$(LLAMA)/ggml/include fnx_mk.cu -o $@ \
	  -L$(BUILD)/bin -lllama -lggml-base -lggml -Xlinker -rpath=$(BUILD)/bin -Xcompiler -fopenmp -lgomp
clean:
	rm -f fnx_mk
fnx_spec: fnx_spec.cu fnx_tables.h
	nvcc -O3 -std=c++17 -arch=$(ARCH) -lineinfo -I$(LLAMA)/include -I$(LLAMA)/ggml/include fnx_spec.cu -o $@ \
	  -L$(BUILD)/bin -lllama -lggml-base -lggml -Xlinker -rpath=$(BUILD)/bin -Xcompiler -fopenmp -lgomp
