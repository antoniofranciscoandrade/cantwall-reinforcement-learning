from gym.envs.registration import register

register(id='FrewEnv-v0',
        entry_point='envs.frew_env:FrewEnv'
)