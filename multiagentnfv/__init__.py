from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentNFVTopo-v0',
    entry_point='multiagentnfv.envs:NFVTopo',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
)

# register(
#     id='MultiagentSimpleSpeakerListener-v0',
#     entry_point='multiagent.envs:SimpleSpeakerListenerEnv',
# )
