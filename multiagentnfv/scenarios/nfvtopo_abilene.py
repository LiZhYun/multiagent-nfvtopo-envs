import numpy as np
from multiagentnfv.core_nfv import World, Agent, Topo, Node, Link
from multiagentnfv.scenario import BaseScenario
import time
from datetime import datetime 
from datetime import timedelta
import sys
import os
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import lil_matrix, coo_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import functools
import copy

np.random.seed(0)

class Scenario(BaseScenario):
    def make_world(self, topo_path, vnf_vocab, usage_vocab):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.index = i
            # agent.collide = True
            # agent.silent = True
            agent.size = 0.15

        
        # world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = False
        #     landmark.movable = False
        # make initial conditions
        self.vnf_vocab = vnf_vocab
        self.usage_vocab = usage_vocab
        self.topo_path = topo_path
        self.reset_world(world)
        return world
    
    def _init_topo(self, topo, topo_path, vnf_vocab, usage_vocab):
        with open(topo_path, 'rb') as f:
            if sys.version_info > (3, 0):
                graph = dict(pkl.load(f, encoding='latin1'))
            else:
                graph = dict(pkl.load(f))
        nodeCap = graph['nodeCap']
        edgeBandwidth = graph['edgeDatarate']
        edgeLatency = graph['edgeLatency']
        nodes = []       
        edges = []
        for node in graph['nodes']:
            new_node = Node()
            new_node.ID = str(node)
            new_node.resource_total = [nodeCap[node], nodeCap[node]]
            new_node.resource_available = [nodeCap[node], nodeCap[node]]
            new_node.resource_cost = [
                np.random.randint(10), np.random.randint(10)]
            new_node.reliability = np.random.random() * 0.15 + 0.8
            new_node.activation_cost = np.random.randint(10)
            new_node.running_cost = np.random.randint(2)
            nodes.append(new_node)
        topo.nodes = nodes         
        topo.node_num = len(nodes)         
        topo.node_num_normal = len(nodes)
        topo.init_hashtable_node()
        for edge in graph['edges']:
            new_link = Link()
            new_link.ID = '(' + str(edge[0]) + str(edge[1]) + ')'
            new_link.bandwidth_total = edgeBandwidth[edge]
            new_link.bandwidth_available = edgeBandwidth[edge]
            new_link.bandwidth_cost = np.random.random() * 0.4 + 0.1
            new_link.link_delay = edgeLatency[edge]
            new_link.in_node = topo.hashtable_node[str(edge[0])]
            new_link.out_node = topo.hashtable_node[str(edge[1])]
            edges.append(new_link)
        
        topo.links = edges
        topo.link_num = len(edges)
        topo.init_hashtable_link()
        vocab = [line.split()[0] for line in open(
            vnf_vocab, 'r', encoding='utf-8', errors='ignore').read().splitlines()]
        usage_vocab = [line.split()[0] for line in open(
            usage_vocab, 'r', encoding='utf-8', errors='ignore').read().splitlines()]
        topo.usage_decoder = {idx: token for idx, token in enumerate(usage_vocab)}
        topo.usage_encoder = {token: idx for idx,
                              token in enumerate(usage_vocab)}
        topo.vnf_decoder = {idx: token for idx, token in enumerate(vocab)}
        topo.vnf_encoder = {token: idx for idx, token in enumerate(vocab)}
        

    def reset_world(self, world):
        # 初始化拓扑
        world.topo = Topo()
        world.topo.ID = 'abilene'
        self._init_topo(world.topo, self.topo_path, self.vnf_vocab, self.usage_vocab)
        self.init_observation(world)
        world.time = datetime.now()  # 世界时间
        world.last_topo = copy.deepcopy(world.topo)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.processing = False
            agent.request = None
            agent.action.subtopo = None
            agent.action.nodes_with_usages = None
            agent.action.links = None
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # agent.state.p_vel = np.zeros(world.dim_p)
            # agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        index = agent.index
        # 约束
        constrain = agent.lambda_unplaced_usage_constrain * world.unplaced_usage_constrain[index] + \
            agent.lambda_optvl_cantfind_constrain * world.optvl_cantfind_constrain[index] + \
            agent.lambda_undefined_usage_constrain * world.undefined_usage_constrain[index] + \
            agent.lambda_in_gress_node_wrong * world.in_gress_node_wrong[index] + \
            agent.lambda_link_cantfind_constrain * world.link_cantfind_constrain[index] + \
            agent.lambda_vnf_embedded_constrain * world.vnf_embedded_constrain[index] + \
            agent.lambda_virtuallink_embedded_constrain * world.virtuallink_embedded_constrain[index] + \
            agent.lambda_bandwidth_constrain * world.bandwidth_constrain[index] + \
            agent.lambda_resource_constrain * world.resource_constrain[index] + \
            agent.lambda_instance_traffic_constrain * world.instance_traffic_constrain[index] + \
            agent.lambda_instance_num_constrain * world.instance_num_constrain[index] + \
            agent.lambda_latency_constrain * world.latency_constrain[index] + \
            agent.lambda_reliability_constrain * world.reliability_constrain[index] + \
            agent.lambda_startend_loc_constrain * world.startend_loc_constrain[index] + \
            agent.lambda_requests_fail_constrain * world.requests_fail_constrain[index] + \
            agent.lambda_topo_fail_constrain * world.topo_fail_constrain[index]

        # 成本
        cost = agent.lambda_instance_startup * world.instance_startup[index] + \
            agent.lambda_instance_expansion * world.instance_expansion[index] + \
            agent.lambda_link_cost * world.link_cost[index] + \
            agent.lambda_node_cost * world.node_cost[index] + \
            agent.lambda_latency * world.latency[index] + \
            agent.lambda_node_activation_cost * world.node_activation_cost[index]
        # 回报
        reward = agent.lambda_requests_received_ratio * world.requests_received_ratio[index] + \
            agent.lambda_requests_accept_reward * world.requests_accept_reward[index] + \
            agent.lambda_vnf_embedded_reward * world.vnf_embedded_reward[index] + \
            agent.lambda_virtuallink_embedded_reward * world.virtuallink_embedded_reward[index] + \
            agent.lambda_bandwidth_reward * world.bandwidth_reward[index] + \
            agent.lambda_resource_reward * world.resource_reward[index] + \
            agent.lambda_latency_reward * world.latency_reward[index] + \
            agent.lambda_reliability_reward * world.reliability_reward[index] + \
            agent.lambda_startend_loc_reward * world.startend_loc_reward[index]
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        rew = reward - cost - constrain
        return rew

    def init_observation(self, world):
        # 从world的topo生成gcn的输入 adj, features, line_adj, edge_features, pm
        edges = world.topo.links
        nodes = world.topo.nodes
        graph_topo = dict()
        for edge in edges:
            if graph_topo.get(int(edge.in_node.ID)) == None:
                graph_topo[int(edge.in_node.ID)] = [int(edge.out_node.ID)]
            else:
                graph_topo[int(edge.in_node.ID)].append(int(edge.out_node.ID))
        for node in nodes:
            if graph_topo.get(int(node.ID)) == None:
                graph_topo[int(node.ID)]=[]
        world.topo.G = nx.from_dict_of_lists(graph_topo)
        for node in world.topo.G:
            world.topo.G.nodes[node]['nodeCap'] = world.topo.hashtable_node[str(node)].resource_available[0]
            world.topo.G.nodes[node]['nodeMem'] = world.topo.hashtable_node[str(node)].resource_available[1]
            world.topo.G.nodes[node]['nodeRel'] = world.topo.hashtable_node[str(node)].reliability
            world.topo.G.nodes[node]['nodeCostCap'] = world.topo.hashtable_node[str(node)].resource_cost[0]
            world.topo.G.nodes[node]['nodeCostMem'] = world.topo.hashtable_node[str(node)].resource_cost[1]
            world.topo.G.nodes[node]['nodeCostRun'] = world.topo.hashtable_node[str(node)].running_cost
            world.topo.G.nodes[node]['nodeCostAct'] = world.topo.hashtable_node[str(node)].activation_cost
            # world.topo.G.nodes[node]['processrequest'] = ' '.join([vnf.request_belong.ID + '_' + vnf.ID for instance in world.topo.hashtable_node[str(
            #     node)].vnf_instances for vnf in instance.vnfs])
            processrequest = ''
            i = 1
            for instance in world.topo.hashtable_node[str(node)].vnf_instances:
                for vnf in instance.vnfs:
                    if i % 3 == 0:
                        processrequest += vnf.request_belong.ID + '_' + vnf.ID + '\n'
                    else:
                        processrequest += vnf.request_belong.ID + '_' + vnf.ID + ' '
                    i += 1
            world.topo.G.nodes[node]['processrequest'] = processrequest
            world.topo.G.nodes[node]['vnf_instances'] = [0] * 17
            for instance in world.topo.hashtable_node[str(node)].vnf_instances:
                world.topo.G.nodes[node]['vnf_instances'][world.topo.vnf_encoder[instance.ID]] = 1
        features = np.array([[world.topo.G.nodes[node]['nodeCap'], world.topo.G.nodes[node]['nodeMem'], world.topo.G.nodes[node]['nodeRel'], world.topo.G.nodes[node]['nodeCostCap'],
                              world.topo.G.nodes[node]['nodeCostMem'], world.topo.G.nodes[node]['nodeCostRun'], world.topo.G.nodes[
            node]['nodeCostAct'], *world.topo.G.nodes[node]['vnf_instances']] for node in list(world.topo.G.nodes)], dtype=np.float)
        features = lil_matrix(features)

        for edge in list(world.topo.G.edges()):
            world.topo.G.edges[edge]['edgeDatarate'] = world.topo.hashtable_link['(' + str(edge[0]) + str(edge[1]) + ')'].bandwidth_available

        world.topo.line_G = nx.line_graph(world.topo.G)
        for node in world.topo.line_G:
            world.topo.line_G.nodes[node]['edgeDatarate'] = world.topo.hashtable_link['(' + str(
                node[0]) + str(node[1]) + ')'].bandwidth_available
            world.topo.line_G.nodes[node]['edgeLatency'] = world.topo.hashtable_link['(' + str(
                node[0]) + str(node[1]) + ')'].link_delay
        edge_l = lil_matrix((len(world.topo.line_G.nodes), 2))
        for i, (u, wt) in enumerate(world.topo.line_G.nodes.data('edgeDatarate')):
            edge_l[i, 0] = wt
        for i, (u, wt) in enumerate(world.topo.line_G.nodes.data('edgeLatency')):
            edge_l[i, 1] = wt
        edge_features = edge_l
        world.topo.line_G.add_edges_from(list(world.topo.line_G.edges), weight=1)

        def _mynode_func(G):
            """Returns a function which returns a sorted node for line graphs.

            When constructing a line graph for undirected graphs, we must normalize
            the ordering of nodes as they appear in the edge.

            """
            if G.is_multigraph():
                def sorted_node(u, v, key):
                    return (u, v, key) if u <= v else (v, u, key)
            else:
                def sorted_node(u, v):
                    return (u, v) if u <= v else (v, u)
            return sorted_node
        degree_array = np.array([world.topo.G.degree[node] for node in world.topo.G])
        degree_variance = np.var(degree_array)
        mysorted_node = _mynode_func(world.topo.G)

        for node in world.topo.G:
            line_edges = [mysorted_node(*x) for x in world.topo.G.edges(node)]

            def comp(a, b):
                if b[0] == a[0]:
                    return a[1] - b[1]
                else:
                    return a[0] - b[0]
            line_edges = sorted(line_edges, key=functools.cmp_to_key(comp))
            # line_edges = [x for x in G.edges(node)]
            if len(line_edges) <= 1:
                continue
            for i, line_edge in enumerate(line_edges):
                for ii, inv_line_edge in enumerate(reversed(line_edges)):
                    if ii < len(line_edges)-1-i and inv_line_edge[0] == line_edge[1] and inv_line_edge[1] != line_edge[0]:
                        world.topo.line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(-np.power(
                            (world.topo.G.degree(inv_line_edge[0]) - 2), 2) / degree_variance)
                    if ii < len(line_edges)-1-i and ((inv_line_edge[0] == line_edge[0] and inv_line_edge[1] != line_edge[1]) or (inv_line_edge[1] == line_edge[1] and inv_line_edge[0] != line_edge[0])):
                        world.topo.line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(
                            -np.power(((world.topo.G.degree(inv_line_edge[1]) + world.topo.G.degree(line_edge[1])) / 2 - 2), 2) / degree_variance)
        line_adj = nx.adjacency_matrix(world.topo.line_G)
        # print(line_adj.todense())
        # graph为字典，键为节点index 值为相连的节点index 先转换为Graph类 再返回邻接矩阵
        adj = nx.adjacency_matrix(world.topo.G)

        world.topo.pm = np.zeros((adj.shape[0], line_adj.shape[0]))

        node_list = list(world.topo.G.nodes)
        edge_list = list(world.topo.line_G.nodes)
        for num, prev in enumerate(node_list):
            for post in range(num+1, len(node_list)):
                if adj[num, post] == 1:
                    edge_index = edge_list.index((prev, node_list[post]))
                    world.topo.pm[num][edge_index] = 1
                    world.topo.pm[post][edge_index] = 1
        
        return adj, features, line_adj, edge_features, world.topo.pm

    def observation(self, agent, world):
        # 从world的topo生成gcn的输入 adj, features, line_adj, edge_features, pm
        # edges = world.topo.links
        # nodes = world.topo.nodes
        # graph_topo = dict()
        # for edge in edges:
        #     if graph_topo.get(int(edge.in_node.ID)) == None:
        #         graph_topo[int(edge.in_node.ID)] = [int(edge.out_node.ID)]
        #     else:
        #         graph_topo[int(edge.in_node.ID)].append(int(edge.out_node.ID))
        # for node in nodes:
        #     if graph_topo.get(int(node.ID)) == None:
        #         graph_topo[int(node.ID)]=[]
        # world.topo.G = nx.from_dict_of_lists(graph_topo)
        for node in world.topo.G:
            world.topo.G.nodes[node]['nodeCap'] = world.topo.hashtable_node[str(node)].resource_available[0]
            world.topo.G.nodes[node]['nodeMem'] = world.topo.hashtable_node[str(node)].resource_available[1]
            world.topo.G.nodes[node]['nodeRel'] = world.topo.hashtable_node[str(node)].reliability
            world.topo.G.nodes[node]['nodeCostCap'] = world.topo.hashtable_node[str(node)].resource_cost[0]
            world.topo.G.nodes[node]['nodeCostMem'] = world.topo.hashtable_node[str(node)].resource_cost[1]
            world.topo.G.nodes[node]['nodeCostRun'] = world.topo.hashtable_node[str(node)].running_cost
            world.topo.G.nodes[node]['nodeCostAct'] = world.topo.hashtable_node[str(node)].activation_cost
            # world.topo.G.nodes[node]['processrequest'] = ' '.join([vnf.request_belong.ID + '_' + vnf.ID for instance in world.topo.hashtable_node[str(
            #     node)].vnf_instances for vnf in instance.vnfs])
            processrequest = ''
            i = 1
            for instance in world.topo.hashtable_node[str(node)].vnf_instances:
                for vnf in instance.vnfs:
                    if i % 3 == 0:
                        processrequest += vnf.request_belong.ID + '_' + vnf.ID + '\n'
                    else:
                        processrequest += vnf.request_belong.ID + '_' + vnf.ID + ' '
                    i += 1
            world.topo.G.nodes[node]['processrequest'] = processrequest
            world.topo.G.nodes[node]['vnf_instances'] = [0] * 17
            for instance in world.topo.hashtable_node[str(node)].vnf_instances:
                world.topo.G.nodes[node]['vnf_instances'][world.topo.vnf_encoder[instance.ID]] = 1
        features = np.array([[world.topo.G.nodes[node]['nodeCap'], world.topo.G.nodes[node]['nodeMem'], world.topo.G.nodes[node]['nodeRel'], world.topo.G.nodes[node]['nodeCostCap'],
                              world.topo.G.nodes[node]['nodeCostMem'], world.topo.G.nodes[node]['nodeCostRun'], world.topo.G.nodes[
            node]['nodeCostAct'], *world.topo.G.nodes[node]['vnf_instances']] for node in list(world.topo.G.nodes)], dtype=np.float)
        features = lil_matrix(features)

        for edge in list(world.topo.G.edges()):
            world.topo.G.edges[edge]['edgeDatarate'] = world.topo.hashtable_link['(' + str(edge[0]) + str(edge[1]) + ')'].bandwidth_available


        # world.topo.line_G = nx.line_graph(world.topo.G)
        for node in world.topo.line_G:
            world.topo.line_G.nodes[node]['edgeDatarate'] = world.topo.hashtable_link['(' + str(node[0]) + str(node[1]) + ')'].bandwidth_available
            world.topo.line_G.nodes[node]['edgeLatency'] = world.topo.hashtable_link['(' + str(
                node[0]) + str(node[1]) + ')'].link_delay
        edge_l = lil_matrix((len(world.topo.line_G.nodes), 2))
        for i, (u, wt) in enumerate(world.topo.line_G.nodes.data('edgeDatarate')):
            edge_l[i, 0] = wt
        for i, (u, wt) in enumerate(world.topo.line_G.nodes.data('edgeLatency')):
            edge_l[i, 1] = wt
        edge_features = edge_l
        # world.topo.line_G.add_edges_from(list(world.topo.line_G.edges), weight=1)

        # def _mynode_func(G):
        #     """Returns a function which returns a sorted node for line graphs.

        #     When constructing a line graph for undirected graphs, we must normalize
        #     the ordering of nodes as they appear in the edge.

        #     """
        #     if G.is_multigraph():
        #         def sorted_node(u, v, key):
        #             return (u, v, key) if u <= v else (v, u, key)
        #     else:
        #         def sorted_node(u, v):
        #             return (u, v) if u <= v else (v, u)
        #     return sorted_node
        # degree_array = np.array([world.topo.G.degree[node] for node in world.topo.G])
        # degree_variance = np.var(degree_array)
        # mysorted_node = _mynode_func(world.topo.G)

        # for node in world.topo.G:
        #     line_edges = [mysorted_node(*x) for x in world.topo.G.edges(node)]

        #     def comp(a, b):
        #         if b[0] == a[0]:
        #             return a[1] - b[1]
        #         else:
        #             return a[0] - b[0]
        #     line_edges = sorted(line_edges, key=functools.cmp_to_key(comp))
        #     # line_edges = [x for x in G.edges(node)]
        #     if len(line_edges) <= 1:
        #         continue
        #     for i, line_edge in enumerate(line_edges):
        #         for ii, inv_line_edge in enumerate(reversed(line_edges)):
        #             if ii < len(line_edges)-1-i and inv_line_edge[0] == line_edge[1] and inv_line_edge[1] != line_edge[0]:
        #                 world.topo.line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(-np.power(
        #                     (world.topo.G.degree(inv_line_edge[0]) - 2), 2) / degree_variance)
        #             if ii < len(line_edges)-1-i and ((inv_line_edge[0] == line_edge[0] and inv_line_edge[1] != line_edge[1]) or (inv_line_edge[1] == line_edge[1] and inv_line_edge[0] != line_edge[0])):
        #                 world.topo.line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(
        #                     -np.power(((world.topo.G.degree(inv_line_edge[1]) + world.topo.G.degree(line_edge[1])) / 2 - 2), 2) / degree_variance)
        line_adj = nx.adjacency_matrix(world.topo.line_G)
        # print(line_adj.todense())
        # graph为字典，键为节点index 值为相连的节点index 先转换为Graph类 再返回邻接矩阵
        adj = nx.adjacency_matrix(world.topo.G)

        # pm = np.zeros((adj.shape[0], line_adj.shape[0]))

        # node_list = list(world.topo.G.nodes)
        # edge_list = list(world.topo.line_G.nodes)
        # for num, prev in enumerate(node_list):
        #     for post in range(num+1, len(node_list)):
        #         if adj[num, post] == 1:
        #             edge_index = edge_list.index((prev, node_list[post]))
        #             pm[num][edge_index] = 1
        #             pm[post][edge_index] = 1
        
        return (adj, features, line_adj, edge_features, world.topo.pm)
    
    # def observation(self, agent, world):
    #     # 从world的topo生成gcn的输入 adj, features, line_adj, edge_features, pm
    #     edges = world.topo.links
    #     nodes = world.topo.nodes
    #     graph_topo = dict()
    #     for edge in edges:
    #         if graph_topo.get(int(edge.in_node.ID)) == None:
    #             graph_topo[int(edge.in_node.ID)] = [int(edge.out_node.ID)]
    #         else:
    #             graph_topo[int(edge.in_node.ID)].append(int(edge.out_node.ID))
    #     for node in nodes:
    #         if graph_topo.get(int(node.ID)) == None:
    #             graph_topo[int(node.ID)] = []
    #     G = nx.from_dict_of_lists(graph_topo)
    #     for node in G:
    #         G.nodes[node]['nodeCap'] = world.topo.hashtable_node[str(
    #             node)].resource_available[0]
    #         G.nodes[node]['nodeMem'] = world.topo.hashtable_node[str(
    #             node)].resource_available[1]
    #         G.nodes[node]['nodeRel'] = world.topo.hashtable_node[str(
    #             node)].reliability
    #         G.nodes[node]['nodeCostCap'] = world.topo.hashtable_node[str(
    #             node)].resource_cost[0]
    #         G.nodes[node]['nodeCostMem'] = world.topo.hashtable_node[str(
    #             node)].resource_cost[1]
    #         G.nodes[node]['nodeCostRun'] = world.topo.hashtable_node[str(
    #             node)].running_cost
    #         G.nodes[node]['nodeCostAct'] = world.topo.hashtable_node[str(
    #             node)].activation_cost
    #         G.nodes[node]['vnf_instances'] = [0] * 17
    #         for instance in world.topo.hashtable_node[str(node)].vnf_instances:
    #             G.nodes[node]['vnf_instances'][world.topo.vnf_encoder[instance.ID]] = 1
    #     features = np.array([[G.nodes[node]['nodeCap'], G.nodes[node]['nodeMem'], G.nodes[node]['nodeRel'], G.nodes[node]['nodeCostCap'],
    #                           G.nodes[node]['nodeCostMem'], G.nodes[node]['nodeCostRun'], G.nodes[
    #         node]['nodeCostAct'], *G.nodes[node]['vnf_instances']] for node in list(G.nodes)], dtype=np.float)
    #     features = lil_matrix(features)

    #     line_G = nx.line_graph(G)
    #     for node in line_G:
    #         line_G.nodes[node]['edgeDatarate'] = world.topo.hashtable_link['(' + str(
    #             edge[0]) + str(edge[1]) + ')'].bandwidth_available
    #         line_G.nodes[node]['edgeLatency'] = world.topo.hashtable_link['(' + str(
    #             edge[0]) + str(edge[1]) + ')'].link_delay
    #     edge_l = lil_matrix((len(line_G.nodes), 2))
    #     for i, (u, wt) in enumerate(line_G.nodes.data('edgeDatarate')):
    #         edge_l[i, 0] = wt
    #     for i, (u, wt) in enumerate(line_G.nodes.data('edgeLatency')):
    #         edge_l[i, 1] = wt
    #     edge_features = edge_l
    #     line_G.add_edges_from(list(line_G.edges), weight=1)

    #     def _mynode_func(G):
    #         """Returns a function which returns a sorted node for line graphs.

    #         When constructing a line graph for undirected graphs, we must normalize
    #         the ordering of nodes as they appear in the edge.

    #         """
    #         if G.is_multigraph():
    #             def sorted_node(u, v, key):
    #                 return (u, v, key) if u <= v else (v, u, key)
    #         else:
    #             def sorted_node(u, v):
    #                 return (u, v) if u <= v else (v, u)
    #         return sorted_node
    #     degree_array = np.array([G.degree[node] for node in G])
    #     degree_variance = np.var(degree_array)
    #     mysorted_node = _mynode_func(G)

    #     for node in G:
    #         line_edges = [mysorted_node(*x) for x in G.edges(node)]

    #         def comp(a, b):
    #             if b[0] == a[0]:
    #                 return a[1] - b[1]
    #             else:
    #                 return a[0] - b[0]
    #         line_edges = sorted(line_edges, key=functools.cmp_to_key(comp))
    #         # line_edges = [x for x in G.edges(node)]
    #         if len(line_edges) <= 1:
    #             continue
    #         for i, line_edge in enumerate(line_edges):
    #             for ii, inv_line_edge in enumerate(reversed(line_edges)):
    #                 if ii < len(line_edges)-1-i and inv_line_edge[0] == line_edge[1] and inv_line_edge[1] != line_edge[0]:
    #                     line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(-np.power(
    #                         (G.degree(inv_line_edge[0]) - 2), 2) / degree_variance)
    #                 if ii < len(line_edges)-1-i and ((inv_line_edge[0] == line_edge[0] and inv_line_edge[1] != line_edge[1]) or (inv_line_edge[1] == line_edge[1] and inv_line_edge[0] != line_edge[0])):
    #                     line_G.edges[line_edge, inv_line_edge]['weight'] = np.exp(
    #                         -np.power(((G.degree(inv_line_edge[1]) + G.degree(line_edge[1])) / 2 - 2), 2) / degree_variance)
    #     line_adj = nx.adjacency_matrix(line_G)
    #     # print(line_adj.todense())
    #     # graph为字典，键为节点index 值为相连的节点index 先转换为Graph类 再返回邻接矩阵
    #     adj = nx.adjacency_matrix(G)

    #     pm = np.zeros((adj.shape[0], line_adj.shape[0]))

    #     node_list = list(G.nodes)
    #     edge_list = list(line_G.nodes)
    #     for num, prev in enumerate(node_list):
    #         for post in range(num+1, len(node_list)):
    #             if adj[num, post] == 1:
    #                 edge_index = edge_list.index((prev, node_list[post]))
    #                 pm[num][edge_index] = 1
    #                 pm[post][edge_index] = 1

    #     return adj, features, line_adj, edge_features, pm
        # entity_pos = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # # communication of all other agents
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent: continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)   

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False



