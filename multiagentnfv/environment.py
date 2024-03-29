import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
# from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(world.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # 设置action space edges_label 12*12 node_label 12*13
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            
            node_action_space = spaces.Box(
                low=-np.inf, high=+np.inf, shape=(len(world.topo.nodes), len(world.topo.usage_decoder)), dtype=np.float32)
            edge_action_space = spaces.Box(
                low=-np.inf, high=+np.inf, shape=(len(world.topo.nodes), len(world.topo.nodes)), dtype=np.float32)
            self.action_space.append([node_action_space, edge_action_space])

            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(12,), dtype=np.float32))
        
        

        self._reset_render()
        # rendering
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n


    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            if agent.processing == False:
                continue
            self._set_action(action_n[i], agent, self.world.topo.G)
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        # reward = np.sum(reward_n)
        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, G):
        agent.action.nodes_with_usages = action[0]
        agent.action.links = action[1]
        agent.action.gen_topo(G)
        
        # # process action
        # if isinstance(action_space, MultiDiscrete):
        #     act = []
        #     size = action_space.high - action_space.low + 1
        #     index = 0
        #     for s in size:
        #         act.append(action[index:(index+s)])
        #         index += s
        #     action = act
        # else:
        #     action = [action]

        # if agent.movable:
        #     # physical action
        #     if self.discrete_action_input:
        #         agent.action.u = np.zeros(self.world.dim_p)
        #         # process discrete action
        #         if action[0] == 1: agent.action.u[0] = -1.0
        #         if action[0] == 2: agent.action.u[0] = +1.0
        #         if action[0] == 3: agent.action.u[1] = -1.0
        #         if action[0] == 4: agent.action.u[1] = +1.0
        #     else:
        #         if self.force_discrete_action:
        #             d = np.argmax(action[0])
        #             action[0][:] = 0.0
        #             action[0][d] = 1.0
        #         if self.discrete_action_space:
        #             agent.action.u[0] += action[0][1] - action[0][2]
        #             agent.action.u[1] += action[0][3] - action[0][4]
        #         else:
        #             agent.action.u = action[0]
        #     sensitivity = 5.0
        #     if agent.accel is not None:
        #         sensitivity = agent.accel
        #     agent.action.u *= sensitivity
        #     action = action[1:]
        # if not agent.silent:
        #     # communication action
        #     if self.discrete_action_input:
        #         agent.action.c = np.zeros(self.world.dim_c)
        #         agent.action.c[action[0]] = 1.0
        #     else:
        #         agent.action.c = action[0]
        #     action = action[1:]
        # # make sure we used all elements of action
        # assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        G = deepcopy(self.world.topo.G)
        elarge = [(u, v) for (u, v, d) in G.edges(
            data=True) if d["edgeDatarate"] > 1500]
        esmall = [(u, v) for (u, v, d) in G.edges(
            data=True) if d["edgeDatarate"] <= 1500]

        if not hasattr(self, 'pos'):
            self.pos = nx.spring_layout(G)  # positions for all nodes
        plt.cla()
        # nodes
        nx.draw_networkx_nodes(G, self.pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, self.pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(
            G, self.pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
        )
        labels = dict()
        for node in G:
            labels[node] = G.nodes[node]['processrequest']
            if labels[node] == []:
                labels[node] = ''
        nx.draw_networkx_labels(
            G, self.pos, labels=labels, font_size=10, font_family="sans-serif")
        plt.axis("off")
        request_accept_num = str(self.world.requests_acceptable)
        request_accept = 'Accepted Request: ' + request_accept_num
        request_reject = str(self.world.requests_rejected)
        request_reject = 'Rejected Request Num: ' + request_reject
        plt.text(-0.7, 1, request_accept)
        plt.text(-0.7, 0.8, request_reject)

        plt.pause(0.1)
        # if mode == 'human':
        #     alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        #     message = ''
        #     for agent in self.world.agents:
        #         comm = []
        #         for other in self.world.agents:
        #             if other is agent: continue
        #             if np.all(other.action == None):
        #                 word = '_'
        #             else:
        #                 word = alphabet
        #             message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
        #     # print(message)

        # for i in range(len(self.viewers)):
        #     # create viewers (if necessary)
        #     if self.viewers[i] is None:
        #         # import rendering only if we need it (and don't import for headless machines)
        #         #from gym.envs.classic_control import rendering
        #         from multiagent import rendering
        #         self.viewers[i] = rendering.Viewer(700,700)

        # # create rendering geometry
        # if self.render_geoms is None:
        #     # import rendering only if we need it (and don't import for headless machines)
        #     #from gym.envs.classic_control import rendering
        #     from multiagent import rendering
        #     self.render_geoms = []
        #     self.render_geoms_xform = []
        #     for entity in self.world.entities:
        #         geom = rendering.make_circle(entity.size)
        #         xform = rendering.Transform()
        #         if 'agent' in entity.name:
        #             geom.set_color(*entity.color, alpha=0.5)
        #         else:
        #             geom.set_color(*entity.color)
        #         geom.add_attr(xform)
        #         self.render_geoms.append(geom)
        #         self.render_geoms_xform.append(xform)

        #     # add geoms to viewer
        #     for viewer in self.viewers:
        #         viewer.geoms = []
        #         for geom in self.render_geoms:
        #             viewer.add_geom(geom)

        # results = []
        # for i in range(len(self.viewers)):
        #     from multiagent import rendering
        #     # update bounds to center around agent
        #     cam_range = 1
        #     if self.shared_viewer:
        #         pos = np.zeros(self.world.dim_p)
        #     else:
        #         pos = self.agents[i].state.p_pos
        #     self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
        #     # update geometry positions
        #     for e, entity in enumerate(self.world.entities):
        #         self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        #     # render to display or array
        #     results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        # return results

    # create receptor field locations in local coordinate frame
    # def _make_receptor_locations(self, agent):
    #     receptor_type = 'polar'
    #     range_min = 0.05 * 2.0
    #     range_max = 1.00
    #     dx = []
    #     # circular receptive field
    #     if receptor_type == 'polar':
    #         for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
    #             for distance in np.linspace(range_min, range_max, 3):
    #                 dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
    #         # add origin
    #         dx.append(np.array([0.0, 0.0]))
    #     # grid receptive field
    #     if receptor_type == 'grid':
    #         for x in np.linspace(-range_max, +range_max, 5):
    #             for y in np.linspace(-range_max, +range_max, 5):
    #                 dx.append(np.array([x,y]))
    #     return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
