#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#####################################################
##           librealsense T265 example             ##
#####################################################

# First import the library
import numpy as np             
import pyrealsense2 as rs 
import inspect


pipeline = rs.pipeline()       
config = rs.config()
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
pipeline.start(config)
i=0  
while i in range(100):
    frames = pipeline.wait_for_frames()
    for frame in frames:
        print((frame.get_profile().stream_name()))
        motion_data = frame.as_motion_frame().get_motion_data()
        
        print(np.array([motion_data.x, motion_data.y, motion_data.z]))
        