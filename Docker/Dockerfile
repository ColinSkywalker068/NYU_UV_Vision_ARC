# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
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

# Install the core packages required to build librealsense libs
RUN apt-get install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
# Install Distribution-specific packages for Ubuntu 18
RUN apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

# Install LibRealSense from source
# We need to build from source because
# the PyPi pip packages are not compatible with Arm processors.
# See link [here](https://github.com/IntelRealSense/librealsense/issues/6964).
	#&& cp config/99-realsense-libusb.rules /etc/udev/rules.d/ \
	#&& cp config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/ \
#	&& udevadm control --reload-rules && udevadm trigger \
# First clone the repository
RUN git clone https://github.com/IntelRealSense/librealsense.git && cd ./librealsense \
	cp config/99-realsense-libusb.rules /etc/udev/rules.d/  && cp config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/ \
	&& udevadm control --reload-rules && udevadm trigger \
	&& mkdir build && cd build && pwd && cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true && make uninstall && make clean && make -j$(nproc)  && make install





## Export pyrealsense2 to your PYTHONPATH so `import pyrealsense2` works
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2


#
# python packages from TF/PyTorch containers
#
COPY python/training/detection/ssd/requirements.txt /tmp/pytorch_ssd_requirements.txt
COPY python/www/flask/requirements.txt /tmp/flask_requirements.txt
COPY python/www/dash/requirements.txt /tmp/dash_requirements.txt

RUN pip3 install --verbose --upgrade Cython && \
    pip3 install --verbose -r /tmp/pytorch_ssd_requirements.txt && \
    pip3 install --verbose -r /tmp/flask_requirements.txt && \
    pip3 install --verbose -r /tmp/dash_requirements.txt


#
# python pip packages
#
RUN pip3 install --no-cache-dir --ignore-installed pybind11 
#RUN pip3 install --no-cache-dir  onnx
RUN pip3 install --no-cache-dir  scipy
RUN pip3 install --no-cache-dir  scikit-learn
RUN pip3 install --no-cache-dir  pandas
RUN pip3 install --no-cache-dir  pycuda
RUN pip3 install --no-cache-dir  numba


#
# CuPy
#
ARG CUPY_VERSION=v9.2.0
ARG CUPY_NVCC_GENERATE_CODE="arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72"

RUN git clone -b ${CUPY_VERSION} --recursive https://github.com/cupy/cupy cupy && \
    cd cupy && \
    pip3 install --no-cache-dir fastrlock && \
    python3 setup.py install --verbose && \
    cd ../ && \
    rm -rf cupy


#
# install OpenCV (with CUDA)
# note:  do this after numba, because this installs TBB and numba complains about old TBB
#
ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

RUN mkdir opencv && \
    cd opencv && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${OPENCV_URL} -O ${OPENCV_DEB} && \
    tar -xzvf ${OPENCV_DEB} && \
    dpkg -i --force-depends *.deb && \
    apt-get update && \
    apt-get install -y -f --no-install-recommends && \
    dpkg -i *.deb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    cd ../ && \
    rm -rf opencv && \
    cp -r /usr/include/opencv4 /usr/local/include/opencv4 && \
    cp -r /usr/lib/python3.6/dist-packages/cv2 /usr/local/lib/python3.6/dist-packages/cv2




#RUN git clone https://github.com/ultralytics/yolov5 && cd yolov5 && pip3 install -r requirements.txt && cd /home/ultraviolet/Downloads
RUN apt-get update && apt-get install -y python3-numpy python3-pillow python3-tk
RUN pip3 install pyserial filterpy tk tqdm seaborn

RUN cd /home/ultraviolet/Downloads
