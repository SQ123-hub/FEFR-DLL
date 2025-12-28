# FEFR-DLL Simulation Environment

This repository contains the simulation environment setup for the paper:
**"FEFR-DLL: Fast and Efficient Fault Recovery Scheme based on Dynamic Link Load in SDN"**

It utilizes **Mininet** for network emulation and **Ryu** as the SDN controller, supporting standard SNDlib topologies (e.g., Germany50, Abilene).

## Prerequisites

* **OS**: Ubuntu 20.04 LTS or 22.04 LTS (Recommended)
* **Python**: 3.8+
* **Mininet**: 2.3.0

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/FEFR-DLL-Simulation.git](https://github.com/YOUR_USERNAME/FEFR-DLL-Simulation.git)
    cd FEFR-DLL-Simulation
    ```

2.  **Run the setup script:**
    This script installs Mininet, Ryu, Python dependencies, and compiles the D-ITG traffic generator.
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```

## Topology Data Preparation

Due to licensing, we cannot distribute the raw topology files. Please download them manually:

1.  Go to the [SNDlib Library](http://sndlib.zib.de/home.action).
2.  Download `germany50.xml` and `abilene.xml` (Native Format).
3.  Place these XML files into the `topologies/` directory.

## Usage

### 1. Start the Controller (Optional Test)
Before running the simulation, ensure you have a Ryu application ready (or use a sample one):
```bash
ryu-manager ryu.app.simple_switch_13