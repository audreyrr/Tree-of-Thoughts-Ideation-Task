ARG CUDA=11.8.0
ARG DIST=ubuntu20.04
ARG TARGET=cudnn8-devel
FROM nvidia/cuda:${CUDA}-${TARGET}-${DIST}

# For automatic installs
ARG DEBIAN_FRONTEND="noninteractive"

# Environment setup
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV HF_HOME=/mnt/cache_volume/hf_cache
ENV TRANSFORMERS_CACHE=/mnt/cache_volume/hf_cache
ENV CONDA_PKGS_DIRS=/conda_pkgs
ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install base tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda3 && \
    rm miniconda.sh

WORKDIR /stage

COPY requirements.txt /stage/requirements.txt
RUN conda init bash && \
    . ~/.bashrc && \
    conda create -y -n test python=3.11 && \
    conda activate test && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /stage/requirements.txt