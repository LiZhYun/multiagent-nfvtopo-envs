import math 
import numpy as np

"""  

返回各种目标  

"""

class Target(object):
    
    # 接收的请求率
    @staticmethod
    def requests_received_ratio(agent, world):
        """接收的请求率"""
        return Target._requests_received_num(agent, world) * (1 / world.requestsnum) * 100

    # 接收的请求数
    @staticmethod 
    def _requests_received_num(agent, world):
        """接收的请求数"""
        return world.requests_acceptable

    # 链路成本
    @staticmethod 
    def link_cost(agent, topo):
        """链路成本"""
        link_cost = 0
        for link in topo.links:
            link_cost += link.bandwidth_occupied * link.bandwidth_cost
        return link_cost
    
    # 节点资源成本
    @staticmethod
    def node_cost(agent, topo):
        """节点资源成本"""
        node_cost = 0
        for node in topo.nodes:
            node_cost += node.cpu_occupied * node.resource_cost[0]
            node_cost += node.mem_occupied * node.resource_cost[1]
        return node_cost
    
    # 延迟
    @staticmethod
    def latency(agent):
        """延迟"""
        layencies = []
        for (a1, a2) in agent.request.data.l_req:
            path_latency = 0  # 一对起点到终点的所有简单路径的延迟
            for path in agent.request.pathPairs[(a1, a2)]:
                single_path_latency = 0  # 一条简单路径上的延迟
                for (u1, u2) in path:
                    pair_latency = 0  # 每对usage的延迟
                    start_name = '_'.join(u1.split('_')[1:])
                    end_name = '_'.join(u2.split('_')[1:])
                    start_vnf = agent.request.hashtable_vnf[start_name]
                    end_vnf = agent.request.hashtable_vnf[end_name]
                    if len(start_vnf.instances_belong) == 0 or len(end_vnf.instances_belong) == 0:
                        layencies.append(1000)
                        continue
                    start_instance = start_vnf.instances_belong[0]
                    end_instance = end_vnf.instances_belong[0]
                    vl = agent.request.hashtable_virtuallink[(
                        start_name, end_name)]
                    for link in vl.links:
                        pair_latency += link.link_delay
                    pair_latency += start_vnf.traffic_required * start_instance.processing_time
                    pair_latency += end_vnf.traffic_required * end_instance.processing_time
                    single_path_latency += pair_latency
                path_latency += single_path_latency
                layencies.append(path_latency)
        return max(layencies)
