import time
import os
import random
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from sndlib_loader import SNDLibTopo


def ensure_dirs():
    if not os.path.exists('data/raw_logs'):
        os.makedirs('data/raw_logs')


def generate_traffic(xml_path, duration=300):

    ensure_dirs()

    topo = SNDLibTopo(xml_path)
    net = Mininet(topo=topo, link=TCLink, switch=OVSKernelSwitch, controller=None)
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
    net.start()

    for sw in net.switches:
        sw.cmd(f'ovs-vsctl set Bridge {sw.name} protocols=OpenFlow13')

    info(f"*** Network started. Warming up for 10 seconds...\n")
    time.sleep(10)

    info("*** Starting ITGRecv on all hosts...\n")
    for h in net.hosts:
        h.cmd(f'cd data/raw_logs && ITGRecv -l recv_log_{h.name}.log &')

    info(f"*** Starting Traffic Generation (Duration: {duration}s)...\n")

    sender_hosts = net.hosts

    for src in sender_hosts:
        dst = random.choice([h for h in net.hosts if h != src])

        cmd = (f'ITGSend -a {dst.IP()} '
               f'-t {duration * 1000} '  # duration in ms
               f'-C 1000 '  # Rate 1000 pkts/sec (Base load)
               f'-c 512 '  # 512 bytes payload
               f'-l send_log_{src.name}_to_{dst.name}.log '  # Log file
               f'&')

        src.cmd(f'cd data/raw_logs && {cmd}')

    info(f"*** Traffic is running... Please wait {duration} seconds.\n")

    for i in range(duration):
        time.sleep(1)
        if i % 10 == 0:
            print(f"Progress: {i}/{duration}s", end='\r')

    info("\n*** Stopping traffic and collecting logs...\n")
    for h in net.hosts:
        h.cmd('killall ITGRecv')
        h.cmd('killall ITGSend')

    net.stop()
    info("*** Experiment finished. Logs saved in data/raw_logs/\n")


if __name__ == '__main__':
    topo_file = 'topologies/germany50.xml'
    generate_traffic(topo_file, duration=300)