# Use an NVIDIA CUDA base image with Ubuntu 20.04 and cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install essential tools and libraries
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    sudo \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libatlas-base-dev \
    gfortran \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-dev python3.12-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.12 get-pip.py && rm get-pip.py

# Ensure that python3 points to python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python3.12 -m pip install --upgrade pip setuptools

# Install Python Machine Learning and Deep Learning libraries
RUN python3.12 -m pip install \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab

# Install PyTorch 2.3.0 with CUDA 11.8 support
RUN python3.12 -m pip install https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp312-cp312-linux_x86_64.whl#sha256=eceaa7ba73df1b0c37d76a5c9eaa1a1ae141512bcae4ed259f2bf06aea591bc3

# Install torchvision compatible with CUDA 11.8 support
RUN python3.12 -m pip install https://download.pytorch.org/whl/cu118/torchvision-0.18.0%2Bcu118-cp312-cp312-linux_x86_64.whl#sha256=30ecda1b67372eeaa68409e242cf28192f71d58d94840ec41c12e7a686ad1747

# Install torchaudio compatible with CUDA 11.8 support
RUN python3.12 -m pip install https://download.pytorch.org/whl/cu118/torchaudio-2.3.0%2Bcu118-cp312-cp312-linux_x86_64.whl#sha256=1d7a4342296f6ef8ad11bfe103378ce0c66a0ffd860105d6f9b863d3c7f8a7f1

# Install additional libraries for transformers and detection models
RUN python3.12 -m pip install \
    opencv-python \
    opencv-contrib-python \
    pillow \
    tqdm \
    h5py \
    requests \
    pyyaml \
    typing-extensions \
    protobuf \
    tensorboard \
    tensorboardX

# Install Hugging Face transformers and other necessary libraries
RUN python3.12 -m pip install transformers timm datasets

# Install additional tools
RUN python3.12 -m pip install \
    rich \
    ipywidgets \
    mypy

# Set the default command to bash
CMD ["/bin/bash"]


