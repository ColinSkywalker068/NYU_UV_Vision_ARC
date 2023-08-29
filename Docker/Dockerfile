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
ARG ROS_PKG=ros_base
ENV ROS_DISTRO=iron
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV CUDA_PATH="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV PATH="/opt/hpcx/ompi/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ARG MAKEFLAGS=-j$(nproc) 

RUN printenv

# change the locale from POSIX to UTF-8
RUN apt-get clean && apt-get update && apt-get install -y locales
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
	  libopenmpi-dev \
          openmpi-bin \
          openmpi-common \
	  protobuf-compiler \
          libprotoc-dev \
		llvm-9 \
          llvm-9-dev nano \
		  lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# stolen from containers/Dockerfile.ros.foxy    
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# 
# install development packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		build-essential \
		cmake \
		git \
		libbullet-dev \
		libpython3-dev \
		python3-colcon-common-extensions \
		python3-flake8 \
		python3-pip \
		python3-numpy \
		python3-pytest-cov \
		python3-rosdep \
		python3-vcstool \
		python3-rosinstall-generator \
		libasio-dev \
		libtinyxml2-dev \
		libcunit1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# install some pip packages needed for testing
RUN python3 -m pip install -U \
		argcomplete \
		flake8-blind-except \
		flake8-builtins \
		flake8-class-newline \
		flake8-comprehensions \
		flake8-deprecated \
		flake8-docstrings \
		flake8-import-order \
		flake8-quotes \
		pytest-repeat \
		pytest-rerunfailures \
		pytest


# Installs librealsense and pyrealsense2 on the Jetson NX running Ubuntu 18.04
# and using Python 3
# Tested on a Jetson NX running Ubuntu 18.04 and Python 3.6.9 on 2020-11-04

RUN apt-get update #&& apt-get -y upgrade
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev && apt-get update && apt-get clean all

# Install the core packages required to build librealsense libs
RUN apt-get install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
# Install Distribution-specific packages for Ubuntu 18
RUN apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libcurl4-openssl-dev

#ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/pyrealsense2/
# stolen from https://github.com/35selim/RealSense-Jetson/blob/main/build_pyrealsense2_and_SDK.sh

# Give the CUDA path to CMake and Set the flags so that librealsense is compatible with python	
RUN	apt-get -y install python3 python3-dev libssl-dev libxinerama-dev libxcursor-dev libcanberra-gtk-module libcanberra-gtk3-module udev 
RUN cd ~ && \
	git clone https://github.com/IntelRealSense/librealsense.git && \
	cd librealsense && mkdir build && cd build && \
	echo "[INFO] pyrealsense2 bindings build has been started!" && \
	sleep 2 && \
	sed -i '3iset(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)\' ../CMakeLists.txt && \
	cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true -DBUILD_NETWORK_DEVICE=ON -DFORCE_RSUSB_BACKEND=ON && \
	make uninstall && make clean && \
	echo "[INFO] Building is starting, it will take a long time like half an hour or more!" && \
	sleep 2 && \
	make -j$(($(nproc)-1)) && make install && \
	echo 'PYTHONPATH="/usr/local/lib:/usr/local/lib/python3.8/pyrealsense2:$PYTHONPATH"' >> ~/.bashrc && \
	echo 'export PYTHONPATH' >> ~/.bashrc 

RUN	cp ~/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d 

RUN /lib/systemd/systemd-udevd --daemon && udevadm control --reload-rules && \
	udevadm trigger && \
	echo "[INFO] pyrealsense2 and RealSense SDK are ready to use!" && \
	echo '[INFO] pyrealsense2 can only be used with "python3" not with "python"!'
	
RUN mkdir /usr/local/lib/python3.8/pyrealsense2
RUN ln -s ~/librealsense/build/wrappers/python/pybackend2.cpython-38-$(uname -m)-linux-gnu.so /usr/local/lib/python3.8/pyrealsense2/
RUN ln -s ~/librealsense/build/wrappers/python/pyrealsense2.cpython-38-$(uname -m)-linux-gnu.so /usr/local/lib/python3.8/pyrealsense2/


#
# python pip packages
#
RUN pip3 install --no-cache-dir --ignore-installed pybind11 
RUN pip3 install --no-cache-dir  onnx
RUN pip3 install --no-cache-dir  scipy
RUN pip3 install --no-cache-dir  scikit-learn
RUN pip3 install --no-cache-dir  pandas
RUN pip3 install --no-cache-dir  pycuda
RUN pip3 install --no-cache-dir  numba
RUN pip3 install --upgrade --no-cache-dir --verbose cmake
RUN cmake --version


#
# CuPy
#
RUN pip3 install --no-cache-dir cupy-wheel 
RUN pip3 install --no-cache-dir --no-deps ultralytics
RUN pip3 install --no-cache-dir py-cpuinfo psutil
RUN pip3 install --no-cache-dir --upgrade matplotlib gitpython
#
# install OpenCV (with CUDA)
# note:  do this after numba, because this installs TBB and numba complains about old TBB
#
ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

RUN cd ~ && \
	mkdir opencv && \
    cd opencv && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${OPENCV_URL} -O ${OPENCV_DEB} && \
    tar -xzvf ${OPENCV_DEB} && \
    dpkg -i --force-depends *.deb && \
    apt-get update && \
    apt-get install -y -f --no-install-recommends && \
    dpkg -i *.deb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
  
RUN cd ../ && \
    rm -rf opencv* 
    #&& \
    #cp -r /usr/include/opencv4 /usr/local/include/opencv4 && \
    #cp -r /usr/lib/python3.8/dist-packages/cv2 /usr/local/lib/python3.8/dist-packages/cv2


RUN apt-get update -y && apt-get install -y python3-numpy python3-pillow python3-tk 
RUN pip3 install --no-cache-dir pyserial filterpy tk seaborn scikit-image tqdm>=4.64.0 setuptools>=65.5.1

#RUN cd /home/ultraviolet/Downloads
#RUN wget https://github.com/ultralytics/yolov5/archive/master.zip -P /root/.cache/torch/hub

#ARG UNZIP_URL=http://ports.ubuntu.com/pool/main/u/unzip/unzip_6.0-25ubuntu1_arm64.deb
#ARG UNZIP_DEB=unzip_6.0-25ubuntu1_arm64.deb
#RUN mkdir unzip && \
#    cd unzip && \
#    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${UNZIP_URL} -O ${UNZIP_DEB} && \
#    dpkg -i --force-depends ${UNZIP_DEB} && \
#    rm -rf /var/lib/apt/lists/* && \
#    apt-get clean && \
#    rm -rf unzip

RUN apt-get install -y unzip

# 
# compile yaml-cpp-0.6, which some ROS packages may use (but is not in the 18.04 apt repo)
#
RUN cd ~ && \
    git clone --branch yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp yaml-cpp-0.6 && \
    cd yaml-cpp-0.6 && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    cp libyaml-cpp.so.0.6.0 /usr/lib/$(uname -m)-linux-gnu/ && \
    ln -s /usr/lib/$(uname -m)-linux-gnu/libyaml-cpp.so.0.6.0 /usr/lib/$(uname -m)-linux-gnu/libyaml-cpp.so.0.6


# 
# download/build ROS from source
#
# https://answers.ros.org/question/325245/minimal-ros2-installation/?answer=325249#post-id-325249
RUN mkdir -p ${ROS_ROOT}/src && \
    cd ${ROS_ROOT} && \
    rosinstall_generator --deps --rosdistro ${ROS_DISTRO} ${ROS_PKG} \
		launch_xml \
		launch_yaml \
		launch_testing \
		launch_testing_ament_cmake \
		demo_nodes_cpp \
		demo_nodes_py \
		example_interfaces \
		camera_calibration_parsers \
		camera_info_manager \
		cv_bridge \
		v4l2_camera \
		vision_opencv \
		vision_msgs \
		image_geometry \
		image_pipeline \
		image_transport \
		compressed_image_transport \
		compressed_depth_image_transport \
		> ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
    cat ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
    vcs import src < ros2.${ROS_DISTRO}.${ROS_PKG}.rosinstall && \
    # https://github.com/dusty-nv/jetson-containers/issues/181
    rm -r ${ROS_ROOT}/src/ament_cmake 

# https://stackoverflow.com/questions/56100217/how-to-accept-the-license-agreement-when-building-rti-connext-dds-5-3-1-with-doc
RUN RTI_NC_LICENSE_ACCEPTED=yes apt-get install -y rti-connext-dds-6.0.1 

RUN git -C ${ROS_ROOT}/src/ clone https://github.com/ament/ament_cmake -b ${ROS_DISTRO} && \
    # install dependencies using rosdep
    cd ${ROS_ROOT} && \
    rosdep init && \
    rosdep update && \
    rosdep install -y \
    	  --ignore-src \
       --from-paths src \
	  --rosdistro ${ROS_DISTRO} \
	  --skip-keys "libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv" 
RUN pip3 install --no-cache-dir  catkin_pkg empy lark
# build it and remove build files
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
	cd ${ROS_ROOT} && \
    colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    rm -rf ${ROS_ROOT}/src && \
    rm -rf ${ROS_ROOT}/logs && \
    rm -rf ${ROS_ROOT}/build && \
    rm ${ROS_ROOT}/*.rosinstall

#
# Set the default DDS middleware to cyclonedds
# https://github.com/ros2/rclcpp/issues/1335
#
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    
COPY ./docker/containers/packages/ros_entrypoint.sh /ros_entrypoint.sh

RUN sed -i \
    's/ros_env_setup="\/opt\/ros\/$ROS_DISTRO\/setup.bash"/ros_env_setup="${ROS_ROOT}\/install\/setup.bash"/g' \
    /ros_entrypoint.sh && \
    cat /ros_entrypoint.sh

RUN echo 'source ${ROS_ROOT}/install/setup.bash' >> /root/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

#RUN cd /root/.cache/torch/hub && unzip ./master.zip && rm ./master.zip
#RUN rm -rf /root/.cache/torch/hub/ultralytics_yolov5_master
# make sure the folder is CV_Detection and not anything else
#WORKDIR /home/ultraviolet/CV_Detection/utils
#CMD python3 main.py
