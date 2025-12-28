#!/bin/bash

# FEFR-DLL Environment Setup Script
# Tested on Ubuntu 20.04 / 22.04

echo "[*] Updating system packages..."
sudo apt-get update

echo "[*] Installing Mininet, Open vSwitch, and Python dependencies..."
sudo apt-get install -y mininet openvswitch-switch python3-pip python3-dev git gcc g++ make

echo "[*] Installing Python libraries for FEFR-DLL..."
pip3 install -r requirements.txt

echo "[*] Installing D-ITG (Traffic Generator)..."
if [ ! -d "D-ITG" ]; then
    git clone https://github.com/AneesManko/D-ITG.git
    cd D-ITG/src
    make
    sudo make install
    cd ../..
    echo "[*] D-ITG installed successfully."
else
    echo "[*] D-ITG directory already exists, skipping clone."
fi

echo "[*] Creating topologies directory..."
mkdir -p topologies

echo "-------------------------------------------------------"
echo "Environment Setup Complete!"
echo "Please download 'germany50.xml' and 'abilene.xml' from"
echo "http://sndlib.zib.de/home.action and place them in the 'topologies/' folder."
echo "-------------------------------------------------------"