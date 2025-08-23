#!/bin/bash
HOME_DIR="/home/ultraviolet"
SCRIPTS_DIR="$HOME_DIR/CV_Detection/services"

# Check if the script is running in the background
IS_BACKGROUND=false
if [[ $(ps -o stat= -p $$) != *+* ]]; then
  IS_BACKGROUND=true
fi

# Switch to ROS2 workspace and start Docker container if running ROS2 command
if [[ ($1 == "run" && $# -ge 3) || $1 == "build" || $1 == "topic" || $1 == "executables" ]]; then
  # Change directory if user is not in ROS2 workspace and this is not ran in the background
  if [[ $IS_BACKGROUND = false ]]; then
    if ! (([ -d src ] && find src -name "package.xml" -print | grep -q .) || find * -maxdepth 1 -name package.xml -print | grep -q .); then
      cd $HOME_DIR/CV_Detection
    fi
  fi
  # Start Docker container if the host does not have ROS2 installed
  source /opt/ros/humble/setup.bash 2> /dev/null || source /opt/ros/humble/install/setup.bash 2> /dev/null
  if [[ $? != 0 ]]; then
    source $SCRIPTS_DIR/start-container.sh "nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-realsense-wrapper" bash -c "cd $(pwd); uv $*; exec bash"
    exit 0
  fi
fi

case $1 in
  # Start Docker container
  "docker")
    # Get the image from the second argument
    image=$2
    if [[ -z "$image" ]]; then
      PS3="Select a Docker image: "
      select type in "Computer Vision" "Navigation (ARM64)" "Navigation (AMD64)" "Quit"
      do
        case $type in
          "Computer Vision")
            image="nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-realsense-wrapper"
            break
            ;;
          "Navigation (ARM64)")
            image="nyuultraviolet/cv-dockerfile:nav-arm64"
            break
            ;;
          "Navigation (AMD64)")
            image="nyuultraviolet/cv-dockerfile:nav-amd64"
            break
            ;;
          "Quit")
            return
            ;;
          *)
            echo "Invalid image. Please try again."
            ;;
          esac
      done
    fi
    # Run container script with the rest of the arguments
    if [[ $# == 1 ]]; then
      shift 1
    else
      shift 2
    fi
    source $SCRIPTS_DIR/start-container.sh $image $@
    ;;
  # Run/Launch ROS2 nodes
  "run")
    # Validate number of parameters
    if (( $# < 2 )); then
      echo "Usage: uv run <package> <node or launch file> <ros args>"
      echo "Example 1: uv run yolov8_ros yolov8_node.py"
      echo "Example 2: uv run teleop keyboard"
      echo "Example 2: uv run uv_utils CCC --ros-args -p kp_wheels:=0.3"
      exit 1
    fi
    # Run setup script
    source install/setup.bash 2> /dev/null
    uv build
    # Find the package
    packages=$(colcon list --names-only)
    if [[ $packages[*] == *$2* ]]; then
      target=$3
      package=$2
      shift 3
    else
      target=$2
      package=$(colcon list --names-only | while read -r pkg; do
        if [[ $target == *.py ]]; then
          checked_path=$(ros2 pkg prefix "$pkg")/share/"$pkg"/launch/"$target"
        else
          checked_path=$(ros2 pkg prefix "$pkg")/lib/"$pkg"/"$target"
        fi
        if [[ -f "$checked_path" ]]; then
            echo "$pkg"
            break
        fi
      done)
      shift 2
    fi
    # Run the nodes
    if [[ $target == *.py ]]; then
      ros2 launch $package $target $@
    else
      ros2 run $package $target $@
    fi
    ;;
  # Build ROS2 packages
  "build")
    if [[ $IS_ROS2_WORKSPACE = true ]]; then
      if (( $# == 2 || $# == 3 )); then
        colcon build --packages-select $2 --cmake-args -DBUILD_TESTING=OFF
      else
        colcon build --cmake-args -DBUILD_TESTING=OFF
      fi
    fi
    if (( $# > 2 )); then
      source install/setup.bash
      package=$2
      target=$3
      shift 3
      if [[ $target == *.py ]]; then
        ros2 launch $package $target $@
      else
        ros2 run $package $target $@
      fi
    fi
    ;;
  # Clean ROS2 build files
  "clean"):
    sudo rm -rf build log install
    ;;
  "topic"):
    source install/setup.bash
    shift 1
    ros2 topic $@
    ;;
  # Update repositories and Docker images
  "update"):
    cd $HOME_DIR/CV_Detection
    git pull
    sudo docker pull nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-realsense-wrapper
    sudo docker pull nyuultraviolet/cv-dockerfile:nav-arm64
    ;;
  # Enable a systemd service
  "enable"):
    if (( $# != 2 )); then
      echo "Usage: uv enable <service_name>"
      echo "service_name: name of the service inside the services folder (ex. uv-can-enable)"
      exit 1
    fi
    sudo cp $2.service /etc/systemd/system/$2.service
    sudo systemctl daemon-reload
    sudo systemctl enable $2.service
    sudo systemctl start $2.service
    ;;
  # Disable a systemd service
  "disable"):
    if (( $# != 2 )); then
      echo "Usage: uv disable <service_name>"
      echo "service_name: name of the service inside the services folder (ex. uv-can-enable)"
      exit 1
    fi
    sudo systemctl disable $2.service
    sudo systemctl stop $2.service
    sudo rm /etc/systemd/system/$2.service
    sudo systemctl daemon-reload
    # Stop docker compose if the service launched a docker stack
    if [[ $2 == "controls-stack" ]]; then
      cd $HOME_DIR/controls_ros/Docker
      sudo docker compose down
    elif [[ $2 == "cv-stack" ]]; then
      cd $HOME_DIR/CV_Detection/Docker
      sudo docker compose down
    fi
    ;;
  # Print usage if command is not recognized
  *):
    echo "Usage: uv <command> <args>"
    echo -e "Docker"
    echo "uv docker: start the system's default Docker container"
    echo "uv docker <image> <docker flags>: start a specific Docker container. <image> can be a docker tag, 'cv', 'controls', or 'microros'"
    echo "sudo docker images: show Docker images"
    echo "sudo docker ps: show Docker containers"
    echo "exit: exits your current Docker container"

    echo -e "\nRun these INSIDE a Docker container"
    echo "uv run <package> <node or launch file> <ros2 args>: run a ROS2 node or launch file. You may leave <package> empty to let uv find your package"
    echo "uv build: build all ROS2 packages in a workspace"
    echo "uv build <package>: build a specific ROS2 package"
    echo "uv build <package> <node or launch file> <ros2 args>: build and run a ROS2 node or launch file"
    echo "uv topic list: view current topics"
    echo "uv topic echo <topic name>: print topic output"

    echo -e "\nRun these OUTSIDE a Docker container"
    echo "uv update: pull latest repositories and images"
    echo "uv enable <service_name>: enables a systemd service"
    echo "uv disable <service_name>: disables a systemd service"
    echo "sudo systemctl status <service_name>.service: checks the status of a systemd service"
    echo "sudo nmtui: configure network"
    echo "sudo ifconfig: view network settings (including IP address)"

    echo -e "\nRun these wherever you want"
    echo "uv clean: remove ROS2 build files from current directory"
    echo "ps -a: view all processes"
    echo "sudo kill <pid>: kill a process"
    echo "nano <file>: edits a file using nano"
    echo "tar -xzvf <source> -C <destination>: unzip a tar archive"
    echo "cat $SCRIPTS_DIR/uv.sh: read more about this command"
    echo -e "\nTo search for something in these instructions: uv | grep <search>"
    ;;
esac
