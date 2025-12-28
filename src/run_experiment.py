import sys
import os
import argparse
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from sndlib_loader import SNDLibTopo


def run_simulation(xml_file, controller_ip='127.0.0.1', controller_port=6633):
    if not os.path.exists(xml_file):
        print(f"Error: Topology file '{xml_file}' not found.")
        sys.exit(1)

    # 1. Initialize Topology
    topo = SNDLibTopo(xml_file)

    net = Mininet(topo=topo,
                  link=TCLink,
                  switch=OVSKernelSwitch,
                  controller=None,
                  autoSetMacs=True)

    info(f"*** Connecting to Remote Controller at {controller_ip}:{controller_port}\n")
    net.addController('c0', controller=RemoteController, ip=controller_ip, port=controller_port)

    info("*** Starting network\n")
    net.start()

    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set Bridge {sw.name} protocols=OpenFlow13')

    info(f"*** Network ready. Topology: {os.path.basename(xml_file)}\n")
    info("*** Type 'exit' or Press Ctrl+D to shut down.\n")

    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FEFR-DLL Mininet Simulation Runner')
    parser.add_argument('--topo', type=str, required=True, help='Path to the SNDlib XML topology file')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Controller IP address')
    args = parser.parse_args()

    setLogLevel('info')
    run_simulation(args.topo, args.ip)