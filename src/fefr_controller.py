from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.lib import hub
import networkx as nx

from fefr_algorithms import FEFRAlgorithms
from gru_model import TrafficGRU, predict_next_step
import torch


class FEFRController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FEFRController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.graph = nx.DiGraph()

        self.algo_engine = None
        self.gru_model = TrafficGRU()
        self.T_outer = 1.97 * 3600
        self.t_inner = 0.79 * 3600
        self.monitor_interval = 5

        self.T_outer /= 100
        self.t_inner /= 100

        self.node_stats = {}
        self.link_types = {}

        self.monitor_thread = hub.spawn(self._monitor)
        self.outer_cycle_thread = hub.spawn(self._outer_cycle_loop)
        self.inner_cycle_thread = hub.spawn(self._inner_cycle_loop)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    def _monitor(self):
        while True:
            for dp in self.dps.values():
                self._request_stats(dp)
            hub.sleep(self.monitor_interval)

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def _outer_cycle_loop(self):
        while True:
            hub.sleep(self.T_outer)
            self.logger.info("[Outer Cycle] Starting Global Optimization...")

            if not self.graph.nodes(): continue

            self.algo_engine = FEFRAlgorithms(self.graph)

            scores = self.algo_engine.get_gnt_bc_scores(self.node_stats)

            self.link_types, link_weights = self.algo_engine.classify_nodes_and_links(scores)

            for (u, v), w in link_weights.items():
                if self.graph.has_edge(u, v):
                    self.graph[u][v]['weight'] = w

            self.logger.info(f"[Outer Cycle] Link Classification Completed. Overloaded links weighted 3.")


    def _inner_cycle_loop(self):
        while True:
            hub.sleep(self.t_inner)
            self.logger.info("[Inner Cycle] Checking for Congestion Risks...")

            threshold = 0.7

            for link in self.graph.edges():
                history = [0.5] * 10
                predicted_load = predict_next_step(self.gru_model, history)

                capacity = 1.0
                utilization = predicted_load / capacity

                if utilization > threshold:
                    self.logger.info(f"[Inner Cycle] Congestion predicted on {link}. Recalculating backup path...")
