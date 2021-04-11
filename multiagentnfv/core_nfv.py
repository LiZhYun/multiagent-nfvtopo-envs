import gc
from functools import reduce
import logging
import numpy as np
import networkx as nx 
import copy
import os,sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from multiagentnfv.nfvparser import Parser
from multiagentnfv.placement_input import PlacementInput
from multiagentnfv.constrains import Constrain
from multiagentnfv.targets import Target
import networkx as nx
from datetime import datetime
# properties and state of physical world entity
logging.basicConfig(level=logging.INFO)
"""

NFV环境核心
包含了 
物理节点、物理链路、拓扑
VNF、VirtualLink、Request、Instance
以及world
执行agent产生的动作，并返回奖励、状态

"""

RES_REQ = {'FW': [0.55, 0.55], 'CACHE': [0.3, 0.3], 'DPI': [0.8, 0.8], 'PCTL': [2.0, 2.0], 'WAPGW': [0.5, 0.5], 'VOPT': [0.7, 0.7], 'HHE': [0.5, 0.5], 'IDS': [0.6, 0.6], 'LB': [
    0.3, 0.3], 'BNG': [0.0, 0.0], 'CR': [0.0, 0.0], 'GGSN': [0.0, 0.0], 'WWW': [0.0, 0.0], 'REG': [0.0, 0.0], 'SRV': [0.0, 0.0], 'PRX': [0.5, 0.5], 'AV': [0.7, 0.7], 'WOPT': [0.6, 0.6], 'IPS': [0.6, 0.6]}
PROCESS_TIME = {'FW': 0.05, 'CACHE': 0.03, 'DPI': 0.08, 'PCTL': 0.02, 'WAPGW': 0.05, 'VOPT': 0.07, 'HHE': 0.05, 'IDS': 0.06, 'LB': 
    0.03, 'BNG': 0, 'CR': 0, 'GGSN': 0, 'WWW': 0, 'REG': 0, 'SRV': 0, 'PRX': 0.05, 'AV': 0.07, 'WOPT': 0.06, 'IPS': 0.06} # 实例处理单位流量的时间
STARTUP_COST = {'FW': 5, 'CACHE': 3, 'DPI': 8, 'PCTL': 2, 'WAPGW': 5, 'VOPT': 7, 'HHE': 5, 'IDS': 6, 'LB': 3, 'BNG': 0.0, 'CR': 0.0, 'GGSN': 0.0, 'WWW': 0.0, 
    'REG': 0.0, 'SRV': 0.0, 'PRX': 5, 'AV': 7, 'WOPT': 6, 'IPS': 6} # 实例启动成本
EXPANSION_COST = {'FW': 0.05, 'CACHE': 0.03, 'DPI': 0.08, 'PCTL': 0.02, 'WAPGW': 0.05, 'VOPT': 0.07, 'HHE': 0.05, 'IDS': 0.06, 'LB': 0.03, 'BNG': 0.0, 'CR': 0.0, 'GGSN': 0.0, 'WWW': 0.0,
                  'REG': 0.0, 'SRV': 0.0, 'PRX': 0.05, 'AV': 0.07, 'WOPT': 0.06, 'IPS': 0.06} # 扩容成本

class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # color
        self.color = None
# 拓扑环境
#物理节点
class Node(Entity): 
    def __init__(self):
        super(Node, self).__init__()
        # 节点ID
        self.ID = None # 0 1
        # 节点总资源 向量(cpu计算，内存)
        self.resource_total = []
        # 节点可用资源
        self.resource_available = []
        # 资源使用成本 向量(cpu计算，内存)
        self.resource_cost = []
        # 节点可靠性
        self.reliability = None
        # 激活成本(第一次在节点新建实例产生)
        self.activation_cost = None
        # 运行成本(节点运行时成本，如果没有实例则进入休眠)
        self.running_cost = None
        # 启动某类vnf实例的成本 （对每类vnf有一项）
        self.vnf_instance_startcost = {}
        # 正在承载的vnf实例集
        self.vnf_instances = []
        # 是否激活(是否运行)
        self.is_activate = False
        # 是否正常工作
        self.working_properly = True
        # 占用的资源
        self.cpu_occupied = 0
        self.mem_occupied = 0
        # 节点上各种实例的有无
        self.instance_type_num = {}
        # 查找字典
        self.hashtable_instances = {}

    def init_hashtable(self):
        for instance in self.vnf_instances:
            self.hashtable_instances[instance.ID] = instance

#物理链路
class Link(Entity):
    def __init__(self):
        super(Link, self).__init__()
        # 链路ID
        self.ID = None # (0,1)
        # 链路总带宽
        self.bandwidth_total = 0
        # 链路可用带宽
        self.bandwidth_available = 0
        # 带宽使用成本
        self.bandwidth_cost = 0
        # 链路时延
        self.link_delay = None
        # 入节点
        self.in_node = None
        # 出节点
        self.out_node = None
        # 正在承载的虚拟链路集
        self.virtual_links = []
        # 是否正常工作         
        self.working_properly = True
        # 被占用的带宽
        self.bandwidth_occupied = 0
        # 查找字典
        self.hashtable_virtual_link = {}

    def init_hashtable(self):
        for virtual_link in self.virtual_links:
            self.hashtable_virtual_link[virtual_link.request_belong.ID + virtual_link.ID] = virtual_link

#物理链路
class Topo(Entity):
    def __init__(self):
        super(Topo, self).__init__()
        # 拓扑ID
        self.ID = None 
        # 节点集合
        self.nodes = []
        # 链路集合
        self.links = []
        # 节点总数
        self.node_num = 0
        # 节点正常数
        self.node_num_normal = 0
        # 链路总数
        self.link_num = 0
        # 链路正常数
        self.link_num_normal = 0
        # 邻接矩阵
        self.adjacenct_matrix = None
        self.G = None
        # usage 解码器
        self.usage_decoder = {}
        self.usage_encoder = {}
        self.vnf_decoder = {}
        self.vnf_encoder = {}
        # 查找字典
        self.hashtable_node = {}
        self.hashtable_link = {}
    
    def init_hashtable(self):
        self.init_hashtable_node()
        self.init_hashtable_link()

    def init_hashtable_node(self):
        for node in self.nodes:
            self.hashtable_node[node.ID] = node

    def init_hashtable_link(self):
        for link in self.links:
            self.hashtable_link[link.ID] = link

# vnf实例
class Instance(Entity):
    def __init__(self):
        super(Instance, self).__init__()
        # 实例ID
        self.ID = None # LB
        # 实例可处理总流量
        self.traffic_total = 0
        # 实例剩余可处理流量
        self.traffic_available = 0
        # 实例类型
        self.type = None
        # 属于的物理节点
        self.node_belong = None
        # 处理单位流量的时间
        self.processing_time = None
        # 启动成本
        self.startup_cost = None
        # 处理的vnf集
        self.vnfs = []
        # 消耗资源数
        self.resource_consume = None
        # 是否休眠
        self.is_sleeping = False
        # 查找字典
        self.hashtable_vnf = {}

    def init_hashtable(self):
        for vnf in self.vnfs:
            self.hashtable_vnf[vnf.request_belong.ID + vnf.ID] = vnf

# VNF
class VNF(Entity):
    def __init__(self):
        super(VNF, self).__init__()
        # vnfID
        self.ID = None # u1_0 u2
        # 所属的请求
        self.request_belong = None
        # 要求的流量
        self.traffic_required = 0
        # 实际到达的流量
        self.traffic_actual = 0
        # vnf占用资源与流量的比例 向量(cpu，内存)
        self.proportion = []
        # 出流量和入流量的比
        self.traffic_ratio = None
        # vnf类型
        self.type = None
        # 是否为备份节点
        self.is_backup = False
        # 由哪些实例处理
        self.instances_belong = []
        # 查找字典
        self.hashtable_instances = {}

    def init_hashtable(self):
        for instance in self.instances_belong:
            self.hashtable_instances[instance.node_belong.ID + instance.ID] = instance

# 虚拟链路
class VirtualLink(Entity):
    def __init__(self):
        super(VirtualLink, self).__init__()
        # 链路ID
        self.ID = None
        # 所属的请求
        self.request_belong = None
        # 要求的数据速率
        self.datarate_required = 0
        # 实际的数据速率
        self.datarate_actual = 0
        # 入vnf
        self.in_vnf = None
        # 出vnf
        self.out_vnf = None
        # 由哪些物理链路承载
        self.links = []
        # 查找字典
        self.hashtable_links = {}

    def init_hashtable(self):
        for link in self.links:
            self.hashtable_links[link.ID] = link

# SFC请求
class Request(Entity):
    def __init__(self):
        super(Request, self).__init__()
        # 请求ID
        self.ID = None # eg req0
        # vnf集
        self.vnfs = []
        # 不确定顺序的vnfs
        # {'req0_u1,req0_u2,req0_u3':['u1','u2']} 不确定usage
        self.optord_vnfs = {}
        # 每种usage的数量
        self.usage_num = {} # {'u1':3, 'u2':1}
        # 虚拟链路集
        self.virtual_links = []
        # 初始数据速率要求
        self.input_datarate = 0
        # 入口节点
        self.ingress = None
        # 出口节点
        self.egress = None
        # 时延要求
        self.delay_requirement  = None
        # 到达时间
        self.arrival_time = None
        # 开始调度事件
        self.schedule_time = None
        # 完成时间         
        self.completion_time = None
        # 经过parser后的请求字典
        self.reqPlacementInput = None
        # 可靠性需求
        self.reliability_requirements = None
        # 是否可接受
        self.is_acceptable = False
        # 完成所需要的时间
        self.time_to_finish = None
        # 原始的request类
        self.request_class = None
        # placementInput类 只是把字典变为类
        self.data = None
        # 每对start,end的所有简单路径
        self.paths = None
        # 每对start,end的所有简单路径中的每一对pairs
        self.pathPairs = None
        
        # 查找字典
        self.hashtable_vnf = {}
        self.hashtable_virtuallink = {}

    def init_hashtable(self):
        self.init_hashtable_vnf()
        self.init_hashtable_virtuallink()

    def init_hashtable_vnf(self):
        for vnf in self.vnfs:
            self.hashtable_vnf[vnf.ID] = vnf

    def init_hashtable_virtuallink(self):
        for virtuallink in self.virtual_links:
            self.hashtable_virtuallink[virtuallink.ID] = virtuallink
        

# action of the agent
class Action(object):
    def __init__(self):
        # 选择的节点及其上承载的请求中的usage
        self.nodes_with_usages = None
        # 选择的物理链路
        self.links = None
        # 生成的子图 node[usagelist]  edges[edgeDatarate]
        self.subtopo = None
        self.origitopo = None

    def gen_topo(self, origitopo):
        self.origitopo = origitopo
        G = nx.DiGraph()
        node_add = []
        a_max = -1
        a_index = 0
        a_row = 0
        if len(self.nodes_with_usages.shape) >= 3:
            self.nodes_with_usages = self.nodes_with_usages.reshape(self.nodes_with_usages.shape[1], self.nodes_with_usages.shape[2])
        if len(self.links.shape) >= 3:
            self.links = self.links.reshape(self.links.shape[1], self.links.shape[2])
        for rindex, row in enumerate(self.nodes_with_usages):
            if row[0] < 0:
                continue
            usagelist = {'usagelist': [0] * 12}
            for cindex, column in enumerate(row[1:]):
                # if cindex == len(row[1:]) - 1:
                #     if a_max < column:
                #         a_max = column
                #         a_index = cindex
                #         a_row = rindex
                if column >= 0.8:
                    usagelist['usagelist'][cindex + 1] = 1
                else:
                    usagelist['usagelist'][cindex + 1] = 0
            if usagelist.get('usagelist') != None:
                node_add.append((rindex, usagelist))

        # for row, column in zip(*np.nonzero(node_labels)):
        #     if usagelist.get(row) == None:
        #         usagelist[row] = []
        #     vnftype = {'vnftype': self.vnf_decoder[node_labels[row]]}
        #     node_add.append((row[0],vnftype))
        G.add_nodes_from(node_add)  # 添加节点
        # if a_row in G:
        #     G.nodes[a_row]['usagelist'][-1] = 1

        # origitopo = self.data[-1]
        gnodes = list(G.nodes)
        for start, end in zip(*np.where(self.links > 0)):
            if start > end and not (start not in gnodes and end not in gnodes) and end in origitopo.adj[start]:
                if start not in gnodes:
                    G.add_nodes_from([(start, {'usagelist': [0] * 12})])
                if end not in gnodes:
                    G.add_nodes_from([(end, {'usagelist': [0] * 12})])
                G.add_edge(int(start), int(end))  # 添加链路
                G.edges[int(start), int(
                    end)]['edgeDatarate'] = self.links[start, end]

        if len(G) == 0:
            G = None
        # if list(nx.isolates(G)) != [] or len(G) == 0 or nx.is_connected(G.to_undirected()) == False:
        #     G = None

        self.subtopo = copy.deepcopy(G)
        del G, node_add, gnodes
        gc.collect()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # 是否正在处理请求
        self.processing = False
        # 正在处理的request
        self.request = None
        # index
        self.index = 0

        # 约束
        self.lambda_unplaced_usage_constrain = 100
        self.lambda_optvl_cantfind_constrain = 100
        self.lambda_undefined_usage_constrain = 1
        self.lambda_in_gress_node_wrong = 100
        self.lambda_link_cantfind_constrain = 100
        self.lambda_vnf_embedded_constrain = 100
        self.lambda_virtuallink_embedded_constrain = 100
        self.lambda_bandwidth_constrain = 10
        self.lambda_resource_constrain = 10
        self.lambda_instance_traffic_constrain = 10
        self.lambda_instance_num_constrain = 10
        self.lambda_latency_constrain = 10
        self.lambda_reliability_constrain = 100
        self.lambda_startend_loc_constrain = 100
        self.lambda_requests_fail_constrain = 100
        self.lambda_topo_fail_constrain = 1000

        # 成本
        self.lambda_instance_startup = 1
        self.lambda_instance_expansion = 10
        self.lambda_link_cost = 1
        self.lambda_node_cost = 1
        self.lambda_latency = 0.1
        self.lambda_node_activation_cost = 1

        # 回报 需要shaping
        self.lambda_requests_received_ratio = 100
        self.lambda_requests_accept_reward = 100
        self.lambda_vnf_embedded_reward = 10  # 正
        self.lambda_virtuallink_embedded_reward = 10  # 正
        self.lambda_bandwidth_reward = 1  # 正
        self.lambda_resource_reward = 1  # 正
        self.lambda_latency_reward = 10  # 正
        self.lambda_reliability_reward = 100  # 正
        self.lambda_startend_loc_reward = 100  # 正
        
        # action
        self.action = Action()

# multi-agent world
class World(object):
    def __init__(self):

        # 包含了agents、拓扑以及请求
        self.agents = []
        self.topo = None
        # 接受的请求
        self.requests_acceptable = 0
        # 拒绝的请求
        self.requests_rejected  = 0
        # 完成的请求
        self.requests_finished = 0
        # 运行的请求         
        self.requests_running = []
        # communication channel dimensionality
        self.dim_c = 0
        # color dimensionality
        self.dim_color = 3
        # 上一个topo状态，用于回滚
        self.last_topo = copy.deepcopy(self.topo)
        # 世界时间
        self.time = None
        
        # 查找字典
        self.hashtable_agent = {}
        self.hashtable_request = {}

    def init_hashtable(self):
        self.init_hashtable_agent()
        self.init_hashtable_request()
        
    def init_hashtable_agent(self):
        for agent in self.agents:
            self.hashtable_agent[agent.ID] = agent
    
    def init_hashtable_request(self):
        for request in self.requests_running:
            self.hashtable_request[request.ID] = request
    # return all entities in the world
    # @property
    # def entities(self):
    #     return self.agents + self.requests + self.topo
    
    @property
    def requestsnum(self):
        return self.requests_acceptable + self.requests_rejected 

    # return all agents controllable by external policies
    # @property
    # def agents(self):
    #     return [agent for agent in self.agents]

    # update state of the world
    def step(self):
        # 约束
        self.unplaced_usage_constrain = [0] * len(self.agents)
        self.optvl_cantfind_constrain = [0] * len(self.agents)
        self.undefined_usage_constrain = [0] * len(self.agents)
        self.in_gress_node_wrong = [0] * len(self.agents)
        self.link_cantfind_constrain = [0] * len(self.agents)
        self.vnf_embedded_constrain = [0] * len(self.agents)
        self.virtuallink_embedded_constrain = [0] * len(self.agents)
        self.bandwidth_constrain = [0] * len(self.agents)
        self.resource_constrain = [0] * len(self.agents)
        self.instance_traffic_constrain = [0] * len(self.agents)
        self.instance_num_constrain = [0] * len(self.agents)
        self.latency_constrain = [0] * len(self.agents)
        self.reliability_constrain = [0] * len(self.agents)
        self.startend_loc_constrain = [0] * len(self.agents)
        self.requests_fail_constrain = [0] * len(self.agents)
        self.topo_fail_constrain = [0] * len(self.agents)

        # 成本
        self.instance_startup = [0] * len(self.agents)
        self.instance_expansion = [0] * len(self.agents)
        self.link_cost = [0] * len(self.agents)
        self.node_cost = [0] * len(self.agents)
        self.latency = [0] * len(self.agents)
        self.node_activation_cost = [0] * len(self.agents)

        # 回报 需要shaping
        self.requests_received_ratio = [0] * len(self.agents)  # 正
        self.requests_accept_reward = [0] * len(self.agents)  # 正
        self.vnf_embedded_reward = [0] * len(self.agents)  # 正
        self.virtuallink_embedded_reward = [0] * len(self.agents)  # 正
        self.bandwidth_reward = [0] * len(self.agents)  # 正
        self.resource_reward = [0] * len(self.agents)  # 正
        self.latency_reward = [0] * len(self.agents)  # 正
        self.reliability_reward = [0] * len(self.agents)  # 正
        self.startend_loc_reward = [0] * len(self.agents)  # 正

        for index, agent in enumerate(self.agents):

            if agent.processing == True and agent.action.subtopo != None:
                logging.info('# processing!')
                self.apply_action(index, agent)

                # vnf约束  所有vnf都需要嵌入
                vnf_embedded_constrain, vnf_embedded_reward = Constrain.vnfembedded(agent)
                self.vnf_embedded_constrain[index] += vnf_embedded_constrain
                self.vnf_embedded_reward[index] += vnf_embedded_reward

                # VirtualLink约束  所有VirtualLink都需要嵌入
                virtuallink_embedded_constrain, virtuallink_embedded_reward = Constrain.virtuallinkembedded(
                    agent)
                self.virtuallink_embedded_constrain[index] += virtuallink_embedded_constrain
                self.virtuallink_embedded_reward[index] += virtuallink_embedded_reward

                # 带宽约束
                bandwidth_constrain, bandwidth_reward = Constrain.bandwidth(
                    agent, self.topo)
                self.bandwidth_constrain[index] += bandwidth_constrain
                self.bandwidth_reward[index] += bandwidth_reward

                # 节点资源约束
                resource_constrain, resource_reward = Constrain.resource(
                    agent, self.topo)
                self.resource_constrain[index] += resource_constrain
                self.resource_reward[index] += resource_reward

                # 实例可处理流量约束
                self.instance_traffic_constrain[index] += Constrain.instance_traffic(
                    agent, self.topo)
                
                # 每个节点上每类实例只能有一个
                self.instance_num_constrain[index] += Constrain.instance_num(
                    agent, self.topo)
                
                # 延迟约束
                latency_constrain, latency_reward = Constrain.latency(agent)
                self.latency_constrain[index] += latency_constrain
                self.latency_reward[index] += latency_reward

                # 可靠性约束
                reliability_constrain, reliability_reward= Constrain.reliability(agent)
                self.reliability_constrain[index] += reliability_constrain
                self.reliability_reward[index] += reliability_reward

                # 起点终点约束
                startend_loc_constrain, startend_loc_reward = Constrain.locofstartend(
                    agent)
                self.startend_loc_constrain[index] += startend_loc_constrain
                self.startend_loc_reward[index] += startend_loc_reward

                
                self.link_cost[index] += Target.link_cost(agent, self.topo)
                self.node_cost[index] += Target.node_cost(agent, self.topo)
                self.latency[index] += Target.latency(agent)

                if self.vnf_embedded_constrain[index] == 0 and self.virtuallink_embedded_constrain[index] == 0 and self.bandwidth_constrain[index] == 0 and self.resource_constrain[index] == 0 and self.startend_loc_constrain[index] == 0:
                    agent.request.is_acceptable = True
                else:
                    agent.request.is_acceptable = False

                if agent.request.is_acceptable == False:  # 请求不能处理 回滚
                    self.requests_fail_constrain[index] += 2
                    self.requests_rejected += 1
                    self.topo = None
                    self.topo = copy.deepcopy(self.last_topo)
                    agent.processing = False

                    del agent.request
                    gc.collect()

                    agent.request = None
                    self.requests_received_ratio[index] += Target.requests_received_ratio(
                        agent, self)
                    logging.info('# Embedded Fail!')

                else:
                    self.requests_accept_reward[index] += 5
                    for node in self.topo.nodes:
                        node.resource_available[0] = node.resource_total[0] -  node.cpu_occupied 
                        node.resource_available[1] = node.resource_total[1] -  node.mem_occupied
                    for link in self.topo.links:
                        link.bandwidth_available = link.bandwidth_total - link.bandwidth_occupied
                    self.last_topo = None
                    self.last_topo = copy.deepcopy(self.topo)
                    agent.request.time_to_finish = self.latency[index]
                    agent.request.schedule_time = datetime.now()
                    self.requests_acceptable += 1
                    self.requests_received_ratio[index] += Target.requests_received_ratio(
                        agent, self)
                    self.requests_running.append(agent.request)
                    agent.processing = False
                    agent.request = None
                    logging.info('# Embedded Successfully!')
                del agent.action.subtopo, agent.action.nodes_with_usages, agent.action.links
                gc.collect()
            elif agent.processing == True and agent.action.subtopo == None:
                logging.info("# Agent %s's action is not legal!" % index)
                self.requests_fail_constrain[index] += 2
                self.topo_fail_constrain[index] += 8
                self.requests_rejected += 1
                self.requests_received_ratio[index] += Target.requests_received_ratio(
                    agent, self)
                agent.processing = False

                del agent.request
                gc.collect()

                agent.request = None
                del agent.action.nodes_with_usages, agent.action.links
                gc.collect()
            else:
                logging.info('# Agent %s is not processing!' % index)
                agent.processing = False
                agent.request = None
            
                    
    def apply_action(self, index, agent):
        # {nodeID:[instance,traffic]}


        nodes_with_usages = agent.action.nodes_with_usages
        links = agent.action.links # {linkID: datarate}
        subtopo = agent.action.subtopo
        self._forceordervnf(index, agent, subtopo)   # 确定了opt usage的顺序
        self._createplacementinput(agent, subtopo)  # 解析最终的request 返回字典 upairs u uf inrate datarate等
        self._createpairspath(agent, subtopo)  # 根据upairs 以及每对起点终点 生成 每条简单路径的usage对 为了后面求latency需要
        self._createvnf(agent, subtopo)  # 根据UF pairs创建vnf类
        self._createvl(agent, subtopo) # 根据U_pairs 创建virtual link类
        self._placeVNF(index, agent, subtopo) # 根据agent的动作 子拓扑 放置vnf 并生成或者扩容instance （包含了备份节点）
        self._placeVL(index, agent, subtopo) #  根据子拓扑 为每条虚拟链路 找到对应的物理链路
        self._calculateresreq_instance(agent, subtopo) # 计算每个instance的资源需求
        self._calculateresreq_link(agent, subtopo)  # 计算每条link的资源需

    def _forceordervnf(self, index, agent, subtopo):  # 确定optvnf的顺序
        
        # ['req0_u1', 'req0_u2', 'req0_u3']
        req_name = agent.request.ID
        for keystr, optusage in agent.request.optord_vnfs.items():
            for i in range(len(optusage) - 1):
                for j in range(len(optusage)-1-i):
                    nodesj = []
                    # nodesj = agent.request.hashtable_vnf[optusage[j]
                    #                                      ].instances_belong
                    indexj = self.topo.usage_encoder[optusage[j]]
                    for node, usagelist in subtopo.nodes.data('usagelist'):
                        if usagelist[indexj] == 1:
                            nodesj.append(node)
                    if len(nodesj) == 0:
                        self.unplaced_usage_constrain[index] += 1
                        continue
                    # nodesj1 = agent.request.hashtable_vnf[optusage[j+1]
                                                        #   ].instances_belong
                    nodesj1 = []
                    # nodesj = agent.request.hashtable_vnf[optusage[j]
                    #                                      ].instances_belong
                    indexj1 = self.topo.usage_encoder[optusage[j+1]]
                    for node, usagelist in subtopo.nodes.data('usagelist'):
                        if usagelist[indexj1] == 1:
                            nodesj1.append(node)
                    if len(nodesj1) == 0:
                        continue
                    for nodej in nodesj:
                        for nodej1 in nodesj1:
                            if nx.has_path(subtopo, nodej, nodej1):
                                tmp = optusage[j]
                                optusage[j] = optusage[j+1]
                                optusage[j+1] = tmp
                                continue
                    self.optvl_cantfind_constrain[index] += 1
            
            
            agent.request.request_class.forceOrder[keystr] = tuple(
                map(lambda usage_name: req_name + '_' + usage_name, optusage))

    def _createplacementinput(self, agent, subtopo):
        prsr = Parser(agent.request.request_class)
        prsr.parse()
        agent.request.reqPlacementInput = prsr.create_pairs()
        # self.placementInput['U_pairs'] = self.pairs   [('req0_a1', 'req0_u1'), ('req0_u1', 'req0_u2_0'), ('req0_u1', 'req0_u2_1')...]
        # self.placementInput['U'] = self.U_out   ['req0_a1', 'req0_u1', 'req0_u2_0', 'req0_u2_1'...]
        # self.placementInput['UF'] = self.UF_out {'req0_a1': 'REG', 'req0_a2': 'SRV'...}
        # self.placementInput['d_req'] = self.d_req {('req0_a1', 'req0_u1'): 1000.0, ('req0_u1', 'req0_u2_0'): 250.0...}
        # self.placementInput['A'] = self.req.A {'req0_a1': 4, 'req0_a2_0': 10, 'req0_a2_1': 7,..}
        # self.placementInput['l_req'] = self.req.l_req {('req0_a1', 'req0_a2'): 10, ('req0_a1', 'req0_a2_0'): 10...}
        # self.placementInput['in_rate'] = self.in_rate {'req0_a1': 1000.0, 'req0_a2': 0, 'req0_a2_0': 480.811...}
        del prsr
        gc.collect()
    
    def _createpairspath(self, agent, subtopo):
        # 每个usage后面要跟的所有usage 为后面找path准备
        nodes = dict()
        agent.request.data = PlacementInput(agent.request.reqPlacementInput)
        for p in agent.request.data.U_pairs:
            if p[0] in nodes:
                nodes[p[0]].append(p[1])
            else:
                nodes[p[0]] = [p[1]]
        for (start, end) in agent.request.data.l_req.keys():
            nodes[end] = []   # 每个usage后面要跟的usage
        # 为每对start end找简单路径
        agent.request.paths = dict()
        for (start, end) in agent.request.data.l_req.keys():
            tmp = []
            tmppaths = []
            self._findAllPathsFrom(start, tmppaths, tmp, nodes)
            # print ("XXX",tmppaths)
            realpaths = []
            for tp in tmppaths:
                if end in tp:
                    realpaths.append(tp)
            #self.paths[(start, end)] = tmppaths
            agent.request.paths[(start, end)] = realpaths
        # 为每对start end的每条简单路径分解为一对一对的usage
        agent.request.pathPairs = dict()
		# print ("PLIST1",self.data.l_req.keys())
        for k in agent.request.data.l_req.keys():
            agent.request.pathPairs[k] = []
            # print ("PLIST2",self.paths[k])
            for p in agent.request.paths[k]:
                ps = self._makePairsFromList(p)
                agent.request.pathPairs[k].append(ps)

        del nodes
        gc.collect()

    def _findAllPathsFrom(self, node, tmppaths, tmp, nodes):
        tmp.append(node)

        if len(nodes[node]) == 0:
            tmppaths.append(tmp)
            return tmppaths
        
        for n in nodes[node]:
            newtmp = list(tmp)
            self._findAllPathsFrom(n, tmppaths, newtmp, nodes)
        #return []
        return tmppaths
    
    def _makePairsFromList(self, plist):
        # print("PLIST", plist)
        ps = []
        for i in range(0, len(plist) - 1):
            ps.append((plist[i], plist[i + 1]))
        return ps
    

    
    def _createvnf(self, agent, subtopo):  # 根据UF pairs创建vnf类
        request_ID = agent.request.ID
        for usage, usage_type in agent.request.data.UF.items():
            vnf_name = '_'.join(usage.split('_')[1:])
            new_vnf = VNF()
            new_vnf.ID = vnf_name
            new_vnf.type = agent.request.data.UF[usage]
            new_vnf.request_belong = agent.request
            new_vnf.proportion = RES_REQ[new_vnf.type]
            new_vnf.traffic_required = agent.request.data.in_rate[usage]
            agent.request.vnfs.append(new_vnf)
            agent.request.hashtable_vnf[new_vnf.ID] = new_vnf

    def _createvl(self, agent, subtopo):
        request_ID = agent.request.ID
        for usage_pair in agent.request.data.U_pairs:
            start_name = '_'.join(usage_pair[0].split('_')[1:])
            end_name = '_'.join(usage_pair[1].split('_')[1:])
            new_virtuallink = VirtualLink()
            new_virtuallink.request_belong = agent.request                                 
            new_virtuallink.in_vnf = agent.request.hashtable_vnf[start_name]
            new_virtuallink.out_vnf = agent.request.hashtable_vnf[end_name]
            new_virtuallink.ID = '(' + start_name + end_name + ')'
            new_virtuallink.datarate_required = agent.request.data.d_req[usage_pair]
            agent.request.virtual_links.append(new_virtuallink)                                 
            agent.request.hashtable_virtuallink[(
                start_name, end_name)] = new_virtuallink   
    
    def _placeVNF(self, agent_index, agent, subtopo):
        # diadj = nx.adjacency_matrix(subtopo) # 有向图的adj  为了表示不确定顺序的usage
        # unsubtopo = subtopo.to_undirected()
        # unadj = nx.adjacency_matrix(unsubtopo) # 无向图的adj 为了表示已确定顺序的usage
        # in_gress_node = None
        # 先放置出入节点
        for a in agent.request.data.A:
            name = '_'.join(a.split('_')[1:])
            vnf = agent.request.hashtable_vnf[name]
            vnf_type = vnf.type
            node = agent.request.data.A[a]

            if vnf_type not in self.topo.hashtable_node[str(node)].hashtable_instances.keys(): # 没有相应实例 则新建
                new_instance = Instance()
                new_instance.ID = vnf_type
                new_instance.type = vnf_type
                new_instance.traffic_total = vnf.traffic_required
                new_instance.processing_time = PROCESS_TIME[vnf_type]
                new_instance.startup_cost = STARTUP_COST[vnf_type]
                new_instance.node_belong = self.topo.hashtable_node[str(node)]
                new_instance.vnfs.append(vnf)  # 将该vnf交给instance处理 资源和流量分配稍后
                new_instance.init_hashtable()
                vnf.instances_belong.append(new_instance)
                vnf.hashtable_instances[new_instance.node_belong.ID + new_instance.ID] = new_instance
                self.topo.hashtable_node[str(node)].instance_type_num[vnf_type] = 1
                self.topo.hashtable_node[str(node)].vnf_instances.append(new_instance)
                self.topo.hashtable_node[str(node)].hashtable_instances[vnf_type] = new_instance  # 实例放入节点上
                # 增加实例启动成本
                self.instance_startup[agent_index] += new_instance.startup_cost
            else: # 存在相应的实例 则append并考虑是否需要重新分配资源
                traffic_available = self.topo.hashtable_node[
                    str(node)].hashtable_instances[vnf_type].traffic_available
                if traffic_available < vnf.traffic_required:
                    expansion = vnf.traffic_required - traffic_available
                    self.topo.hashtable_node[str(node)].hashtable_instances[
                        vnf_type].traffic_total += expansion
                    self.topo.hashtable_node[str(node)].hashtable_instances[
                        vnf_type].traffic_available = 0
                    # 增加扩容成本
                    self.instance_expansion[agent_index] += expansion * \
                        EXPANSION_COST[vnf_type]
                instance = self.topo.hashtable_node[str(node)].hashtable_instances[vnf_type]
                instance.vnfs.append(vnf)
                instance.hashtable_vnf[vnf.request_belong.ID + vnf.ID] = vnf
                vnf.instances_belong.append(instance)
                vnf.hashtable_instances[instance.node_belong.ID +
                                        instance.ID] = instance

        visited_usage = {} # 为了u1_0等usage

        for node, usagelist in subtopo.nodes.data('usagelist'):  # usagelist为13长的list 0代表end 1-11代表u1-u11 12代表a1
            for index, usage in enumerate(usagelist):
                
                if usage == 1:
                    # if index == len(usagelist) - 1: # a1 只需要确定好起点
                        # in_gress_node = node
                        # if in_gress_node != agent.request.data.A[agent.request.ID + '_a1']:
                        #     self.in_gress_node_wrong[agent_index] += 1
                    # 其他需要判断是否node上有对应的实例
                    usage_name = self.topo.usage_decoder[index] # u1
                    if visited_usage.get(usage_name) == None: # 第一次遍历到u1
                        visited_usage[usage_name] = 0
                    else:
                        visited_usage[usage_name] += 1
                    # hashtable_vnf的key为request中 u1_0,u2等等，若u1不在其中，则u1_0肯定在
                    # usage_name_0 = usage_name + '_0'
                    # 遍历到了不属于request的usage 违反了约束
                    # print(agent.request.hashtable_vnf.keys)
                    if usage_name in agent.request.hashtable_vnf.keys() and visited_usage[usage_name] > 0: # 为了本来只有u1 但是有多个的 进行创建vnf
                        # usage_name = usage_name + '_' + \
                        #     str(visited_usage[usage_name])
                        # new_vnf = copy.deepcopy(
                        #     agent.request.hashtable_vnf[self.topo.usage_decoder[index]])
                        # new_vnf.ID = usage_name
                        # new_vnf.is_backup = True
                        # agent.request.vnfs.append(new_vnf)
                        # agent.request.hashtable_vnf[usage_name] = new_vnf
                        vnf = agent.request.hashtable_vnf[usage_name]
                        vnf.is_backup = True
                    if usage_name not in agent.request.hashtable_vnf.keys(): # u1_0
                        usage_name = usage_name + \
                            '_' + str(visited_usage[usage_name])
                        if usage_name not in agent.request.hashtable_vnf.keys():
                            if self.topo.usage_decoder[index] + '_0' not in agent.request.hashtable_vnf.keys():
                                self.undefined_usage_constrain[agent_index] += 1
                                continue
                            else: # 为如 u2_5 超过了本身需要的usage新建vnf 为了可靠性
                                vnf_nums = 0
                                for vnf_name in agent.request.hashtable_vnf.keys():
                                    if self.topo.usage_decoder[index] in vnf_name:
                                        vnf_nums += 1
                                randnum = np.random.randint(vnf_nums)
                                usage_name = self.topo.usage_decoder[index] + '_' + str(randnum)
                                vnf = agent.request.hashtable_vnf[usage_name]
                                vnf.is_backup = True
                                # new_vnf = copy.deepcopy(agent.request.hashtable_vnf[self.topo.usage_decoder[index] + '_0'])
                                # new_vnf.ID = usage_name
                                # new_vnf.is_backup = True
                                # agent.request.vnfs.append(new_vnf)
                                # agent.request.hashtable_vnf[usage_name] = new_vnf
                    
                            
                    vnf = agent.request.hashtable_vnf[usage_name]
                    vnf_type = vnf.type
                    if len(self.topo.hashtable_node[str(node)].vnf_instances) == 0:
                        self.node_activation_cost[agent_index] += self.topo.hashtable_node[str(node)].activation_cost
                    if vnf_type not in self.topo.hashtable_node[str(node)].hashtable_instances.keys(): # 没有相应实例 则新建
                        new_instance = Instance()
                        new_instance.ID = vnf_type
                        new_instance.type = vnf_type
                        new_instance.traffic_total = vnf.traffic_required
                        new_instance.processing_time = PROCESS_TIME[vnf_type]
                        new_instance.startup_cost = STARTUP_COST[vnf_type]
                        new_instance.node_belong = self.topo.hashtable_node[str(node)]
                        new_instance.vnfs.append(vnf)  # 将该vnf交给instance处理 资源和流量分配稍后
                        new_instance.init_hashtable()
                        vnf.instances_belong.append(new_instance)
                        vnf.hashtable_instances[new_instance.node_belong.ID + new_instance.ID] = new_instance
                        self.topo.hashtable_node[str(node)].instance_type_num[vnf_type] = 1
                        self.topo.hashtable_node[str(node)].vnf_instances.append(new_instance)
                        self.topo.hashtable_node[str(node)].hashtable_instances[vnf_type] = new_instance  # 实例放入节点上
                        # 增加实例启动成本
                        self.instance_startup[agent_index] += new_instance.startup_cost
                    else: # 存在相应的实例 则append并考虑是否需要重新分配资源
                        traffic_available = self.topo.hashtable_node[
                            str(node)].hashtable_instances[vnf_type].traffic_available
                        if traffic_available < vnf.traffic_required:
                            expansion = vnf.traffic_required - traffic_available
                            self.topo.hashtable_node[str(node)].hashtable_instances[
                                vnf_type].traffic_total += expansion
                            self.topo.hashtable_node[str(node)].hashtable_instances[
                                vnf_type].traffic_available = 0
                            # 增加扩容成本
                            self.instance_expansion[agent_index] += expansion * \
                                EXPANSION_COST[vnf_type]
                        instance = self.topo.hashtable_node[str(node)].hashtable_instances[vnf_type]
                        instance.vnfs.append(vnf)
                        instance.hashtable_vnf[vnf.request_belong.ID + vnf.ID] = vnf
                        vnf.instances_belong.append(instance)
                        vnf.hashtable_instances[instance.node_belong.ID +
                                                instance.ID] = instance
                    
    def _placeVL(self, index, agent, subtopo):
        subtopo = subtopo.to_undirected()
        # self._createoptVL(agent, subtopo)  # 为optusage创建vl
        origitopo = agent.action.origitopo
        for vl in agent.request.virtual_links:

            startvnf = vl.in_vnf
            endvnf = vl.out_vnf
            if len(startvnf.instances_belong) == 0:
                # logging.info('startvnf not embedded')
                self.unplaced_usage_constrain[index] += 1
                if len(endvnf.instances_belong) == 0:
                    # logging.info('endvnf not embedded')
                    self.unplaced_usage_constrain[index] += 1
                continue
            if len(endvnf.instances_belong) == 0:
                # logging.info('endvnf not embedded')
                self.unplaced_usage_constrain[index] += 1
                continue
            startnode = int(startvnf.instances_belong[0].node_belong.ID)
            endnode = int(endvnf.instances_belong[0].node_belong.ID)
            if startvnf.ID.startswith('a') or endvnf.ID.startswith('a'):
                if nx.has_path(origitopo, startnode, endnode) == False:
                    self.link_cantfind_constrain[index] += 1
                    continue
                path_nodes = nx.shortest_path(origitopo, source=startnode, target=endnode)
                for i in range(len(path_nodes) - 1):
                    link = self.topo.hashtable_link['(' + str(path_nodes[i]) + str(path_nodes[i+1]) + ')']
                    vl.links.append(link)
                    vl.hashtable_links[link.ID] = link
                    link.virtual_links.append(vl)
                    link.hashtable_virtual_link[vl.request_belong.ID + vl.ID] = vl
            else:
                if nx.has_path(subtopo, startnode, endnode) == False:
                    self.link_cantfind_constrain[index] += 1
                    continue
                path_nodes = nx.shortest_path(subtopo, source=startnode, target=endnode)
                for i in range(len(path_nodes) - 1):
                    link = self.topo.hashtable_link['(' + str(path_nodes[i]) + str(path_nodes[i+1]) + ')']
                    vl.links.append(link)
                    vl.hashtable_links[link.ID] = link
                    link.virtual_links.append(vl)
                    link.hashtable_virtual_link[vl.request_belong.ID + vl.ID] = vl
        del subtopo
        gc.collect()

    def _calculateresreq_instance(self, agent, subtopo):  # 计算每个instance的资源需求 cpu和存储
        # for u in agent.request.data.U:  # req0_u1_0
        #     # u1
        #     usage = ''.join(u.split('_')[1:])
        #     agent.request.resource_req[u] = agent.request.data.in_rate[u] * \
        #         agent.request.hashtable_vnf[usage].proportion
        for node in self.topo.nodes:
            for instance in node.vnf_instances:
                # instance.resource_consume = instance.traffic_total * RES_REQ[instance.type]
                res_consume = RES_REQ[instance.type][:]
                instance.resource_consume = list(map(
                    lambda consume: consume * instance.traffic_total, res_consume))
                # instance.resource_consume = reduce(
                #     lambda cpu, mem: cpu * instance.traffic_total + mem * instance.traffic_total, RES_REQ[instance.type])

    # 计算每条link的资源需求 带宽
    def _calculateresreq_link(self, agent, subtopo):
        for link in self.topo.links:
            for vl in link.virtual_links:
                link.bandwidth_occupied += vl.datarate_required
        


    # # integrate physical state
    # def integrate_state(self, p_force):
    #     for i,entity in enumerate(self.entities):
    #         if not entity.movable: continue
    #         entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
    #         if (p_force[i] is not None):
    #             entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
    #         if entity.max_speed is not None:
    #             speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
    #             if speed > entity.max_speed:
    #                 entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
    #                                                               np.square(entity.state.p_vel[1])) * entity.max_speed
    #         entity.state.p_pos += entity.state.p_vel * self.dt     

# self.unplaced_usage_constrain = 0
        # self.optvl_cantfind_constrain = 0

        # wait_for_add_post = False
        # last_prev_post = None
        # for optusage, prev_post in agent.request.optord_vnfs.items():
        #     prev = prev_post[0]
        #     post = prev_post[1]

        #     for i in range(len(optusage) - 1):
        #         for j in range(len(optusage)-1-i):
        #             nodesj = []
        #             # nodesj = agent.request.hashtable_vnf[optusage[j]
        #             #                                      ].instances_belong
        #             indexj = self.topo.usage_encoder[optusage[j]]
        #             for node, usagelist in subtopo.nodes.data('usagelist'):
        #                 if usagelist[indexj] == 1:
        #                     nodesj.append(node)
        #             if nodesj == []:
        #                 self.unplaced_usage_constrain += 1
        #                 continue
        #             # nodesj1 = agent.request.hashtable_vnf[optusage[j+1]
        #                 #   ].instances_belong
        #             nodesj1 = []
        #             # nodesj = agent.request.hashtable_vnf[optusage[j]
        #             #                                      ].instances_belong
        #             indexj1 = self.topo.usage_encoder[optusage[j+1]]
        #             for node, usagelist in subtopo.nodes.data('usagelist'):
        #                 if usagelist[indexj1] == 1:
        #                     nodesj1.append(node)
        #             if nodesj1 == []:
        #                 continue
        #             for nodej in nodesj:
        #                 for nodej1 in nodesj1:
        #                     if nx.has_path(subtopo, nodej, nodej1):
        #                         tmp = optusage[j]
        #                         optusage[j] = optusage[j+1]
        #                         optusage[j+1] = tmp
        #                         continue
        #             self.optvl_cantfind_constrain += 1
        #     for i, usage in enumerate(optusage):
        #         prefix = 'req' + agent.request.ID
        #         if prefix + usage in agent.request.reqPlacementInput['parallel_num'].keys:
        #             copynum = agent.request.reqPlacementInput['parallel_num'][prefix + usage]
        #             for j in range(copynum):
        #                 if i != len(optusage) - 1:
        #                     for index in range(i+1, len(optusage)-1):
        #                         new_vnf = copy.deepcopy(
        #                             agent.request.hashtable_vnf[usage])
        #                         new_vnf.ID = optusage[index] + '_' + str(j)
        #                         agent.request.vnfs.append(new_vnf)
        #                         agent.request.hashtable_vnf[optusage[index] +
        #                                                     '_' + str(j)] = new_vnf
        #             agent.request.vnfs.remove(
        #                 agent.request.hashtable_vnf[usage])
        #             del agent.request.hashtable_vnf[usage]
        # if wait_for_add_post == True:
        #     agent.request.optord_vnfs[last_prev_post][1] = optusage[0]
        #     agent.request.optord_vnfs[optusage][0] = agent.request.optord_vnfs[last_prev_post][0]
        # if post == '':
        #     wait_for_add_post = True
        #     last_prev_post = optusage

    # def _createvl(self, agent, subtopo):
    #     # 为optusage创建VL
    #     for optusage, prev_post in agent.request.optord_vnfs.items():
    #         for i, usage in enumerate(optusage):
    #             prefix = 'req' + agent.request.ID
    #             if prefix + usage in agent.request.reqPlacementInput['parallel_num'].keys:
    #                 copynum = agent.request.reqPlacementInput['parallel_num'][prefix + usage]
    #                 for j in range(copynum):
    #                     if i != len(optusage) - 1:
    #                         for index in range(i+1, len(optusage)-1):
    #                             new_virtuallink = VirtualLink()
    #                             new_virtuallink.request_belong = agent.request
    #                             new_virtuallink.in_vnf = agent.request.hashtable_vnf[optusage[index] + '_' + str(
    #                                 j)]
    #                             new_virtuallink.out_vnf = agent.request.hashtable_vnf[optusage[index+1] + '_' + str(
    #                                 j)]
    #                             new_virtuallink.ID = (
    #                                 optusage[index], optusage[index+1])
    #                             agent.request.virtual_links.append(
    #                                 new_virtuallink)
    #                             agent.request.hashtable_virtuallink[(
    #                                 optusage[index], optusage[index+1])] = new_virtuallink
    #                         new_virtuallink = VirtualLink()   # 为分支opt的最后一个usage与后面的加上link
    #                         new_virtuallink.request_belong = agent.request
    #                         new_virtuallink.in_vnf = agent.request.hashtable_vnf[optusage[index]]
    #                         new_virtuallink.out_vnf = agent.request.hashtable_vnf[prev_post[1]]
    #                         new_virtuallink.ID = (
    #                             optusage[index], prev_post[1])
    #                         agent.request.virtual_links.append(
    #                             new_virtuallink)
    #                         agent.request.hashtable_virtuallink[(
    #                             optusage[index], prev_post[1])] = new_virtuallink
    #                     else:  # 最后一个为分支usage 则只需要与后继链接
    #                         new_virtuallink = VirtualLink()   # 为分支opt的最后一个usage与后面的加上link
    #                         new_virtuallink.request_belong = agent.request
    #                         new_virtuallink.in_vnf = agent.request.hashtable_vnf[optusage[i]]
    #                         new_virtuallink.out_vnf = agent.request.hashtable_vnf[prev_post[1]]
    #                         new_virtuallink.ID = (
    #                             optusage[i], prev_post[1])
    #                         agent.request.virtual_links.append(
    #                             new_virtuallink)
    #                         agent.request.hashtable_virtuallink[(
    #                             optusage[i], prev_post[1])] = new_virtuallink
    #                 break  # 分支usage后的链路已经全部处理完

    #             new_virtuallink = VirtualLink()
    #             new_virtuallink.request_belong = agent.request
    #             new_virtuallink.in_vnf = agent.request.hashtable_vnf[optusage[i]]
    #             new_virtuallink.out_vnf = agent.request.hashtable_vnf[optusage[i+1]]
    #             new_virtuallink.ID = (
    #                 optusage[i], optusage[i+1])
    #             agent.request.virtual_links.append(new_virtuallink)
    #             agent.request.hashtable_virtuallink[(
    #                 optusage[i], optusage[i+1])] = new_virtuallink
    #         new_virtuallink = VirtualLink()  # 前驱
    #         new_virtuallink.request_belong = agent.request
    #         new_virtuallink.in_vnf = agent.request.hashtable_vnf[prev_post[0]]
    #         new_virtuallink.out_vnf = agent.request.hashtable_vnf[optusage[0]]
    #         new_virtuallink.ID = (
    #             prev_post[0], optusage[0])
    #         agent.request.virtual_links.append(new_virtuallink)
    #         agent.request.hashtable_virtuallink[(
    #             prev_post[0], optusage[0])] = new_virtuallink
    #         new_virtuallink = VirtualLink()  # 后继
    #         new_virtuallink.request_belong = agent.request
    #         new_virtuallink.in_vnf = agent.request.hashtable_vnf[optusage[-1]]
    #         new_virtuallink.out_vnf = agent.request.hashtable_vnf[prev_post[1]]
    #         new_virtuallink.ID = (
    #             optusage[-1], prev_post[1])
    #         agent.request.virtual_links.append(new_virtuallink)
    #         agent.request.hashtable_virtuallink[(
    #             optusage[-1], prev_post[1])] = new_virtuallink
