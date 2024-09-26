# Start with a base image that includes CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.0.1 with CUDA 11.8
RUN pip3 install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN ln -s /usr/bin/python3 /usr/bin/python

# clone NVIDIA apex repository
WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/apex.git

# checkout specific commit
WORKDIR /workspace/apex
RUN git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
RUN pip3 install -r requirements.txt
RUN python setup.py install --cpp_ext --cuda_ext
WORKDIR /workspace