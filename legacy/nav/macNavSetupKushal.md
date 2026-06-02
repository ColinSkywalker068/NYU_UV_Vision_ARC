# RoboMaster Docker Quick Start

## First Time Setup

```bash
cd ~/nyu/CV_Detection
docker run -it -p 5901:5901 -p 6080:6080 --name robomaster-nav -v $(pwd):/workspace nyuultraviolet/cv-dockerfile:nav-arm64 bash

# Inside container:
apt-get update && apt-get install -y tigervnc-standalone-server novnc websockify xfce4 xfce4-goodies dbus-x11
vncpasswd
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
source /opt/ros/humble/setup.bash
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup
```

## Every Time

```bash
# Start container (via Docker Desktop or):
docker start robomaster-nav
docker exec -it robomaster-nav bash

# Inside container - start VNC:
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
websockify -D --web=/usr/share/novnc/ 6080 localhost:5901
```

**Open browser:** http://localhost:6080/vnc.html

## Usage

- **GUI apps (RViz2):** Run in VNC desktop browser
- **CLI commands:** Use `docker exec -it robomaster-nav bash` in new terminal (faster)

## Troubleshooting

```bash
# If ports busy:
pkill -9 Xtigervnc websockify
```
