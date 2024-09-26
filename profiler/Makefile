#
# Copyright 2011-2018 NVIDIA Corporation. All rights reserved
# 
INCLUDES = -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

# Set required library paths. 
# In the case of cross-compilation, set the libs to the correct ones under /usr/local/cuda/targets/<TARGET_ARCH>-<TARGET_OS>/lib

ifeq ($(OS), Windows_NT)
    LIB_PATH ?= /usr/local/cuda/extras/CUPTI/lib64
else
    ifneq ($(TARGET_ARCH), $(HOST_ARCH))
        INCLUDES += -I/usr/local/cuda/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
        LIB_PATH ?= /usr/local/cuda/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
        TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
    else
        EXTRAS_LIB_PATH := /usr/local/cuda/extras/CUPTI/lib64
        LIB_PATH ?= /usr/local/cuda/lib64
    endif
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(LIB_PATH)
    LIBS= -lcuda -L $(LIB_PATH) -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS= -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti
    else
        LIBS :=
        ifeq ($(HOST_ARCH), $(TARGET_ARCH))
            export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
            LIBS = -L $(EXTRAS_LIB_PATH)
        endif
        LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti
    endif
    OBJ = o
endif

# Point to the necessary cross-compiler.

ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++
        endif
    endif
    
    ifdef HOST_COMPILER
        NVCC_COMPILER = -ccbin $(HOST_COMPILER)
    endif
endif

activity_trace_async: activity_trace_async.$(OBJ) vec.$(OBJ)
	nvcc $(NVCC_COMPILER) -o $@ $^ $(LIBS) $(INCLUDES)

activity_trace_async.$(OBJ): activity_trace_async.cpp
	nvcc $(NVCC_COMPILER) -c $(INCLUDES) $<

vec.$(OBJ): vec.cu
	nvcc $(NVCC_COMPILER) -c $(INCLUDES) $<

run: activity_trace_async
	./$<

clean:
	rm -rf activity_trace_async activity_trace_async.$(OBJ) vec.$(OBJ) build dist vtrain_profiler.egg-info
