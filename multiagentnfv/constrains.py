import math
import numpy as np

"""

返回各种约束项

"""

class Constrain(object):

    # vnf约束  所有vnf都需要嵌入
    @staticmethod
    def vnfembedded(agent):
        """所有vnf都需要嵌入"""
        vnf_embedded_constrain = 0
        vnf_embedded_reward = 0
        for vnf in agent.request.vnfs:
            if vnf.instances_belong == []:
                vnf_embedded_constrain += 1
            else:
                vnf_embedded_reward += 1
        return vnf_embedded_constrain, vnf_embedded_reward
    
    # VirtualLink约束  所有VirtualLink都需要嵌入
    @staticmethod          
    def virtuallinkembedded(agent):
        """所有VirtualLink都需要嵌入"""
        virtuallink_embedded_constrain = 0
        virtuallink_embedded_reward = 0
        for virtuallink in agent.request.virtual_links:
            if virtuallink.links == []:
                virtuallink_embedded_constrain += 1
            else:
                virtuallink_embedded_reward += 1
        return virtuallink_embedded_constrain, virtuallink_embedded_reward

    # 带宽约束
    @staticmethod
    def bandwidth(agent, topo):
        """带宽约束"""
        bandwidth_constrain = 0
        bandwidth_reward = 0
        for link in topo.links:
            constrain_degree = link.bandwidth_occupied - link.bandwidth_total if link.bandwidth_occupied - link.bandwidth_total > 0 else 0
            bandwidth_constrain += constrain_degree
            if constrain_degree == 0:
                bandwidth_reward += 1
        return bandwidth_constrain, bandwidth_reward
    
    # 节点资源约束
    @staticmethod 
    def resource(agent, topo):
        """节点资源约束"""
        resource_constrain = 0
        resource_reward = 0
        for node in topo.nodes:
            for instance in node.vnf_instances:
                node.cpu_occupied += instance.resource_consume[0]
                node.mem_occupied += instance.resource_consume[1]
            cpu_degree = node.cpu_occupied - node.resource_total[0] if node.cpu_occupied - node.resource_total[0] > 0 else 0
            mem_degree = node.mem_occupied - node.resource_total[1] if node.mem_occupied - node.resource_total[1] > 0 else 0
            if cpu_degree == 0:
                resource_reward += 1
            if mem_degree == 0:
                resource_reward += 1
            resource_constrain += cpu_degree + mem_degree
             
        return resource_constrain, resource_reward
    
    # 实例可处理流量约束     
    @staticmethod      
    def instance_traffic(agent, topo):
        """实例可处理流量约束"""
        instance_traffic_constrain = 0
        for node in topo.nodes:
            for instance in node.vnf_instances:
                constrain_degree = abs(instance.traffic_available) if instance.traffic_available < 0 else 0
                instance_traffic_constrain += constrain_degree
        return instance_traffic_constrain

    # 每个节点上每类实例只能有一个       
    @staticmethod           
    def instance_num(agent, topo):
        """每个节点上每类实例只能有一个"""
        instance_num_constrain = 0
        for node in topo.nodes:
            num_degree = 0
            for vnftype, num in node.instance_type_num.items():
                if num > 1:
                    num_degree += num - 1
            instance_num_constrain += num_degree
        return instance_num_constrain
    
    # 延迟约束            
    @staticmethod                
    def latency(agent):
        """延迟约束"""
        latency_constrain = 0
        latency_reward = 0
        for (a1, a2) in agent.request.data.l_req:
            path_latency = 0 # 一对起点到终点的所有简单路径的延迟
            for path in agent.request.pathPairs[(a1, a2)]:
                single_path_latency = 0 # 一条简单路径上的延迟
                for (u1, u2) in path:
                    pair_latency = 0 # 每对usage的延迟
                    start_name = '_'.join(u1.split('_')[1:])
                    end_name = '_'.join(u2.split('_')[1:])
                    start_vnf = agent.request.hashtable_vnf[start_name]
                    end_vnf = agent.request.hashtable_vnf[end_name]
                    if len(start_vnf.instances_belong) == 0 or len(end_vnf.instances_belong) == 0:
                        latency_constrain += agent.request.data.l_req[(a1, a2)]
                        continue
                    start_instance = start_vnf.instances_belong[0]
                    end_instance = end_vnf.instances_belong[0]
                    vl = agent.request.hashtable_virtuallink[(start_name, end_name)]
                    for link in vl.links:
                        pair_latency += link.link_delay
                    pair_latency += start_vnf.traffic_required * start_instance.processing_time
                    pair_latency += end_vnf.traffic_required * end_instance.processing_time
                    single_path_latency += pair_latency
                path_latency += single_path_latency
                if path_latency == 0:
                    latency_constrain += 100
                    continue
                constrain_degree = path_latency - agent.request.data.l_req[(a1, a2)] if path_latency - agent.request.data.l_req[(a1, a2)] > 0 else 0
                if constrain_degree == 0:
                    latency_reward += 1
                latency_constrain += constrain_degree
        return latency_constrain, latency_reward
            
    # 可靠性约束                 
    @staticmethod                     
    def reliability(agent):
        """可靠性约束"""
        reliability_constrain = 0
        reliability_reward = 0
        total_reli = 1
        for vnf in agent.request.vnfs:
            vnf_invalid = 1
            for instance in vnf.instances_belong:
                node = instance.node_belong
                node_reli = node.reliability
                vnf_invalid *= (1 - node_reli)
            vnf_reli = 1 - vnf_invalid
            total_reli *= vnf_reli
        constrain_degree = agent.request.reliability_requirements - total_reli if agent.request.reliability_requirements - total_reli > 0 else 0
        if constrain_degree == 0:
            reliability_reward += 1
        reliability_constrain += constrain_degree
        return reliability_constrain, reliability_reward

    # 起点终点约束                      
    @staticmethod                          
    def locofstartend(agent):
        """起点终点位置约束"""
        startend_loc_constrain = 0
        startend_loc_reward = 0
        for a in agent.request.data.A:
            name = '_'.join(a.split('_')[1:])
            vnf = agent.request.hashtable_vnf[name]
            if len(vnf.instances_belong) == 0:
                startend_loc_constrain += 1
                continue
            node = vnf.instances_belong[0].node_belong
            if int(node.ID) != int(agent.request.data.A[a]):
                startend_loc_constrain += 1
            else:
                startend_loc_reward += 1
        return startend_loc_constrain, startend_loc_reward

        
            
