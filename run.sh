#!/bin/bash
# A script to run a command sequence inside a Docker container at startup.

# Give the system (especially Docker daemon and network) a few seconds to initialize.
sleep 15

# --- Commands to run INSIDE the Docker Container ---
# We group the commands that must run inside the container into a single string.
# The '&&' ensures that the next command only runs if the previous one was successful.
COMMANDS_IN_DOCKER="uv clean && uv build && uv run yolo_cpp realsense.launch.py"

# --- Execute the Commands ---
# This line assumes 'uv docker' starts/enters a container and executes a command
# passed to it. We use `/bin/bash -c "..."` which is a standard and robust
# way to run a string of shell commands inside the container's environment.
uv docker /bin/bash -c "${COMMANDS_IN_DOCKER}"
