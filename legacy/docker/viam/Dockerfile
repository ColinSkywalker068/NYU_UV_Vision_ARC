ARG BASE_IMAGE=python:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

WORKDIR /jetson-inference

#
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ARG MAKEFLAGS=-j$(nproc) 

RUN printenv

# change the locale from POSIX to UTF-8
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

#
# apt packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          python3-pip \
		  python3-dev \
          python3-matplotlib \
              python-is-python3 \
		  build-essential \
		  gfortran \
		  git \
		  cmake \
		  curl \
		  libopenblas-dev \
		  liblapack-dev \
		  libblas-dev \
		  libhdf5-serial-dev \
		  hdf5-tools \
		  libhdf5-dev \
		  zlib1g-dev \
		  zip \
		  libjpeg8-dev \
		  libopenmpi2 \
          openmpi-bin \
          openmpi-common \
		  protobuf-compiler \
          libprotoc-dev \
		llvm-9 \
          llvm-9-dev nano \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean




# Installs librealsense and pyrealsense2 on the Jetson NX running Ubuntu 18.04
# and using Python 3
# Tested on a Jetson NX running Ubuntu 18.04 and Python 3.6.9 on 2020-11-04

RUN apt-get update #&& apt-get -y upgrade
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-setuptools \
    python3-pip \
    python3-dev && apt-get update && apt-get install -y lsb-release && apt-get clean all

RUN pip3 install viam-server pyserial flask flask_cors flask_restful 

# make sure the folder is CV_Detection and not anything else
WORKDIR /home/ultraviolet/CV_Detection/utils
