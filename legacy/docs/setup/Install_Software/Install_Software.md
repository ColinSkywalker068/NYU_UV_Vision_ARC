# Install Software

## Install PyRealsense

1\. Install nano with 

`sudo apt install nano`

2\. To configure CUDA run

`nano ~./bashrc`

then go down to the end of the file using the arrow keys and add
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```
to the end of the file. Finally, apply the changes by running

`source ~/.bashrc`

3\. Run CMake

`cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true`

4\. Run make

`make -j4`

5\. Install

` make install`

6\.

## Install CV System

1\. Clone this repo
`git clone 