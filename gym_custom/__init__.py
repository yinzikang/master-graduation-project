from gym.envs.registration import register
register(
    id='TrainEnvVariableStiffness-v6',
    entry_point='gym_custom.envs:TrainEnvVariableStiffnessV6',
)
register(
    id='TrainEnvVariableStiffnessAndPosture-v6',
    entry_point='gym_custom.envs:TrainEnvVariableStiffnessAndPostureV6',
)
register(
    id='TrainEnvVariableStiffness-v7',
    entry_point='gym_custom.envs:TrainEnvVariableStiffnessV7',
)
register(
    id='TrainEnvVariableStiffnessAndPosture-v7',
    entry_point='gym_custom.envs:TrainEnvVariableStiffnessAndPostureV7',
)
