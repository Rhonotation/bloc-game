#!/bin/bash

# 1. Kill any leftover display servers from previous runs
echo "Cleaning up old sessions..."
killall Xvfb x11vnc novnc_proxy 2>/dev/null

# 2. Set the display environment variable
export DISPLAY=:1

# 3. Start the virtual frame buffer (virtual screen)
echo "Starting virtual display (Xvfb)..."
Xvfb :1 -screen 0 1024x768x16 &
sleep 1 # Give it a second to initialize

# 4. Start the VNC server to capture the virtual display
echo "Starting VNC server..."
x11vnc -display :1 -nopw -listen localhost -xkb &
sleep 1

# 5. Start the noVNC proxy to make it web-accessible
echo "Starting noVNC proxy on port 6080..."
./noVNC/utils/novnc_proxy --vnc localhost:5900 --listen 6080 &
sleep 1

# 6. Run your Python script
echo "Launching bloc.py..."
python bloc.py