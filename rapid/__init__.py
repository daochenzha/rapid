from gym.envs.registration import register

register(
    id='EpisodeInvertedPendulum-v2',
    entry_point='rapid.mujoco_envs:EpisodeInvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='EpisodeSwimmer-v2',
    entry_point='rapid.mujoco_envs:EpisodeSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='DensityEpisodeSwimmer-v2',
    entry_point='rapid.mujoco_envs:DensityEpisodeSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='ViscosityEpisodeSwimmer-v2',
    entry_point='rapid.mujoco_envs:ViscosityEpisodeSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='EpisodeWalker2d-v2',
    max_episode_steps=1000,
    entry_point='rapid.mujoco_envs:EpisodeWalker2dEnv',
)

register(
    id='EpisodeHopper-v2',
    entry_point='rapid.mujoco_envs:EpisodeHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

