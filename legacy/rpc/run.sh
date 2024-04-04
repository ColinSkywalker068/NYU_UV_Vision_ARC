#!/bin/sh
cd /home/ultraviolet/CV_Detection/Docker/viam/

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
/usr/bin/docker-compose -f /home/ultraviolet/CV_Detection/Docker/viam/docker-compose.yml up $@