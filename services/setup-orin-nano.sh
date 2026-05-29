#!/bin/bash
# Clone repositories
cd ~
git clone git@gitlab.com:nyu-robomaster/CV_Detection.git

# Install basic packages
sudo apt-get install -y can-utils busybox nano vim

# Configure CAN registers
sudo busybox devmem 0x0c303018 w 0xc458
sudo busybox devmem 0x0c303010 w 0xc400

# Start systemd services on boot
cd ~/controls_ros
sudo cp ../services/ethernet.service /etc/systemd/system/ethernet.service
sudo systemctl daemon-reload
sudo systemctl enable ethernet.service
sudo systemctl start ethernet.service

# Disable sudo password by modifying sudoers file
RULE="$USER ALL=(ALL) NOPASSWD: ALL"
if ! sudo grep -q "$RULE" /etc/sudoers; then
  echo "$RULE" | sudo EDITOR='tee -a' visudo
fi

# Add uv as a shell command
sudo ln -s "/home/ultraviolet/CV_Detection/services/uv.sh" /usr/local/bin/uv

# Install Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Pull Docker images
sudo docker pull nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus
sudo docker pull nyuultraviolet/cv-dockerfile:nav-arm64
