#!/bin/bash
# Max power and fan for Jetson Orin

# Set max power mode (mode 0 = MAXN)
sudo nvpmodel -m 0

# Set max clocks
sudo jetson_clocks

# Max fan speed (if pwm-fan exists)
if [ -d /sys/devices/pwm-fan ]; then
    echo 255 | sudo tee /sys/devices/pwm-fan/target_pwm > /dev/null
fi

echo "Power and fan set to maximum"

