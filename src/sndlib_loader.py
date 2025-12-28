import xml.etree.ElementTree as ET
from mininet.topo import Topo
from mininet.link import TCLink


class SNDLibTopo(Topo):

    def build(self, xml_path, bandwidth=1000):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            raise RuntimeError(f"Failed to parse XML file {xml_path}: {e}")

        ns = {'n': 'http://sndlib.zib.de/network'}

        if 'network' not in root.tag:
            ns = {}

        print(f"[*] Loading topology from {xml_path}...")

        nodes_map = {}

        nodes = root.findall('.//n:node', ns) if ns else root.findall('.//node')

        for node in nodes:
            node_id = node.get('id')

            x_elem = node.find('n:coordinates/n:x', ns) if ns else node.find('coordinates/x')
            y_elem = node.find('n:coordinates/n:y', ns) if ns else node.find('coordinates/y')

            node_opts = {}
            if x_elem is not None and y_elem is not None:
                node_opts['x'] = float(x_elem.text)
                node_opts['y'] = float(y_elem.text)

            sw = self.addSwitch(node_id, **node_opts)
            nodes_map[node_id] = sw

            h = self.addHost(f'h_{node_id}')
            self.addLink(sw, h, cls=TCLink, bw=1000, delay='0ms')

        links = root.findall('.//n:link', ns) if ns else root.findall('.//link')

        for link in links:
            src_id = link.find('n:source', ns).text if ns else link.find('source').text
            target_id = link.find('n:target', ns).text if ns else link.find('target').text

            if src_id in nodes_map and target_id in nodes_map:
                self.addLink(nodes_map[src_id],
                             nodes_map[target_id],
                             cls=TCLink,
                             bw=bandwidth,
                             delay='5ms')
            else:
                print(f"[!] Warning: Link references unknown nodes: {src_id} -> {target_id}")

        print(f"[*] Topology loaded: {len(nodes_map)} switches and {len(links)} links created.")