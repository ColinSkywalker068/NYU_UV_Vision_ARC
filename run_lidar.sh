cat << 'EOF' > /home/ultraviolet/CV_Detection/run_lidar.sh
#!/bin/bash

### -----------------------------------------
### 1. SET PATHS
### -----------------------------------------
WS=/home/ultraviolet/lidar_ws
SRC=$WS/src
PKG=$SRC/sllidar_ros2

### -----------------------------------------
### 2. ENSURE WORKSPACE EXISTS
### -----------------------------------------
mkdir -p "$SRC"

### -----------------------------------------
### 3. COPY DRIVER IF NOT ALREADY PRESENT
### -----------------------------------------
if [ ! -d "$PKG" ]; then
    echo "[INFO] Copying sllidar_ros2 into workspace..."
    cp -r /home/ultraviolet/lidar/src/sllidar_ros2 "$SRC/"
fi

### -----------------------------------------
### 4. CLEAN + BUILD WORKSPACE
### -----------------------------------------
echo "[INFO] Cleaning old build..."
rm -rf "$WS/build" "$WS/install" "$WS/log"

echo "[INFO] Building workspace..."
source /opt/ros/humble/setup.bash
cd "$WS"
colcon build --symlink-install
if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed."
    exit 1
fi

### -----------------------------------------
### 5. SOURCE WORKSPACE
### -----------------------------------------
source "$WS/install/local_setup.bash"

### -----------------------------------------
### 6. AUTO-DETECT LIDAR SERIAL PORT
### -----------------------------------------
echo "[INFO] Detecting /dev/ttyUSB* devices..."
PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -n 1)

if [ -z "$PORT" ]; then
    echo "[ERROR] No /dev/ttyUSB* device found."
    echo "Is the LIDAR plugged in? Is the USB device passed into Docker?"
    exit 1
fi

echo "[INFO] Using LIDAR on: $PORT"

### -----------------------------------------
### 7. FIX PERMISSIONS
### -----------------------------------------
sudo chmod 666 "$PORT"

### -----------------------------------------
### 8. LAUNCH LIDAR DRIVER (A2M12)
### -----------------------------------------
echo "[INFO] Starting LIDAR driver..."
ros2 launch sllidar_ros2 view_sllidar_a2m12_launch.py \
    serial_port:="$PORT" \
    serial_baudrate:=256000

EOF