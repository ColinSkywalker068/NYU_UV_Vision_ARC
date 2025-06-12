#!/bin/bash

uv build yolov8_ros launch.py && uv run yolov8_ros launch.py

echo "Script finished."