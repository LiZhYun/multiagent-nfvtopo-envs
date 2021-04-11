import logging  
logging.basicConfig(level=logging.INFO)
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# sys.path.append(os.path.join(sys.path[0], '..'))

from copy import deepcopy
import argparse
import numpy as np
from request_test import requestlist, create_request, process_request
from multiagentnfv.environment import MultiAgentEnv
from multiagentnfv.policy import InteractivePolicy
import multiagentnfv.scenarios as scenarios
import time
from datetime import datetime
import threading

REQ_NUM = 0


def release_resources(lock, world):
    while True:
        # logging.info("Releasing Resource!")
        lock.acquire()
        world.time = datetime.now()
        for request in world.requests_running[:]:
            if (world.time - request.schedule_time).seconds >= request.time_to_finish:
                logging.info("# Releasing Resource!")
                world.requests_running.remove(request)
                world.requests_finished.append(request)
                for vl in request.virtual_links:
                    for link in vl.links:
                        link.bandwidth_occupied -= vl.datarate_required
                        link.bandwidth_available = link.bandwidth_total - link.bandwidth_occupied
                for vnf in request.vnfs:
                    for instance in vnf.instances_belong:
                        instance.traffic_available += vnf.traffic_required
                        ratio = np.random.randint(2)
                        alpha = 0.8
                        origin = instance.traffic_available
                        instance.traffic_available = instance.traffic_available * \
                            alpha * ratio + \
                            instance.traffic_available * (1 - ratio)
                        instance.traffic_total -= (origin -
                                                instance.traffic_available)

        time.sleep(0.01)
        lock.release()

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='nfvtopo_abilene.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world(
        "multiagentnfv/scenarios/topo/netGraph_abilene.pickle", "multiagentnfv/scenarios/request/vnf.vocab", "multiagentnfv/scenarios/request/usage.vocab")
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    obs_n = env.reset()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    request_list = requestlist()
    request_n = []
    for i in range(env.n):
        request_num = np.random.randint(len(request_list))
        request_n.append(deepcopy(request_list[request_num]))
    REQ_NUM = process_request(request_n, REQ_NUM)
    create_request(request_n, env.agents)
    # execution loop
    lock = threading.RLock()
    t = threading.Thread(target=release_resources, args=(lock, env.world))
    t.setDaemon(True)
    t.start()
    while True:
        # query for action from each agent's policy
        logging.info("# Request Num %s" % str(REQ_NUM))
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        if np.any(reward_n):
            logging.info("# Reward is {}".format(reward_n))
        # render all agent views
        env.render()
        request_n = []
        for i in range(env.n):
            request_num = np.random.randint(len(request_list))
            request_n.append(deepcopy(request_list[request_num]))
        REQ_NUM = process_request(request_n, REQ_NUM)
        create_request(request_n, env.agents)
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
