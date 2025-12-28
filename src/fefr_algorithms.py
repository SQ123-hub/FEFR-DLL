import networkx as nx
import numpy as np
import math


class FEFRAlgorithms:
    def __init__(self, nx_graph):
        self.graph = nx_graph

    def calculate_entropy(self, flows):
        total_bw = sum(flows)
        if total_bw == 0:
            return 0
        entropy = 0
        for f in flows:
            p_i = f / total_bw
            if p_i > 0:
                entropy -= p_i * math.log(p_i, 2)
        return entropy

    def get_gnt_bc_scores(self, node_traffic_history):
        [cite_start]
        epsilon = 0.5  # [cite: 195]
        scores = {}

        bc_raw = nx.betweenness_centrality(self.graph, weight='weight', normalized=True)

        bc_sum = sum(bc_raw.values())
        p_bc = {k: v / bc_sum if bc_sum > 0 else 0 for k, v in bc_raw.items()}

        for node in self.graph.nodes():
            if node not in node_traffic_history:
                gnt = 0
            else:
                data = node_traffic_history[node]
                flows_t1 = data.get('prev', [])
                flows_t2 = data.get('curr', [])

                delta_p = abs(sum(flows_t2) - sum(flows_t1))
                delta_h = abs(self.calculate_entropy(flows_t2) - self.calculate_entropy(flows_t1))

                gnt = delta_p / delta_h if delta_h > 1e-6 else 0

            scores[node] = epsilon * gnt + (1 - epsilon) * p_bc.get(node, 0)

        return scores

    def classify_nodes_and_links(self, node_scores):
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_nodes)

        idx_a = int(n * 0.2)
        idx_b = int(n * 0.7)  # 20% + 50%

        overloaded = set(node for node, score in sorted_nodes[:idx_a])
        normal = set(node for node, score in sorted_nodes[idx_a:idx_b])
        idle = set(node for node, score in sorted_nodes[idx_b:])

        link_weights = {}
        link_types = {}

        for u, v in self.graph.edges():
            is_ov_u = u in overloaded
            is_ov_v = v in overloaded
            is_idle_u = u in idle
            is_idle_v = v in idle

            if is_ov_u and is_ov_v:
                l_type = 'Overloaded'
                weight = 3
            elif (is_idle_u and is_idle_v) or (is_idle_u and v in normal) or (u in normal and is_idle_v):
                l_type = 'Idle'
                weight = 1
            else:
                l_type = 'Normal'
                weight = 2

            link_weights[(u, v)] = weight
            link_types[(u, v)] = l_type

        return link_types, link_weights

    def dbps_algorithm(self, src, dst, link_types, k=2):
        temp_graph = self.graph.copy()
        edges_to_remove = [e for e, t in link_types.items() if t == 'Idle']

        temp_graph.remove_edges_from(edges_to_remove)

        try:
            paths = list(nx.islice(nx.shortest_simple_paths(temp_graph, src, dst, weight='weight'), k))
        except nx.NetworkXNoPath:
            try:
                paths = list(nx.islice(nx.shortest_simple_paths(self.graph, src, dst, weight='weight'), k))
            except nx.NetworkXNoPath:
                return []

        return paths