#!/bin/bash


echo "[*] Cleaning up previous runs..."
sudo mn -c

echo "[*] Starting Ryu Controller (Background)..."
ryu-manager src/fefr_controller.py --verbose &
RYU_PID=$!

sleep 5

echo "[*] Starting Mininet Topology (Germany50)..."
sudo python3 src/run_experiment.py --topo topologies/germany50.xml --ip 127.0.0.1

echo "[*] Experiment Complete. Killing Controller..."
kill $RYU_PID