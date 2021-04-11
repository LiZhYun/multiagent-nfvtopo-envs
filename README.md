# Multi-Agent NFV Environment

为以下仓库所写的实验环境，参考了以下论文及仓库:
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).


**贡献:** 根据NFV环境需要，根据Gym自行构建了适合NFV的多智能体环境。 

## 安装

- 首先`cd` 到项目根目录， 并在cmd中输入`pip install -e .`(安装项目中的包)

- 实验环境: Python (3.6), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

- 这样可以保证multiagent-nfv-envs在您的搜索路径中

## 运行

- 在项目根目录执行 `python  bin/interactive.py`

### 创建您自己的拓扑环境

您可以根据自己的拓扑需求创建新的Scenario (`make_world()`, `reset_world()`, `reward()`, `observation()`).

## Paper citation

<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>
