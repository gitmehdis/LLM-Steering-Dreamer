import gym
import crafter
import numpy as np

class GymAPIWrapper(gym.Wrapper):
    """Wrapper to handle old/new Gym API compatibility."""
    def __init__(self, env, seed=None):
        super().__init__(env)
        self._seed = seed
        if seed is not None:
            self.seed(seed)

    def seed(self, seed):
        """Set the seed for the environment."""
        self._seed = seed
        np.random.seed(seed)
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        if self._seed is not None:
            self.seed(self._seed)
        obs = self.env.reset()
        if not isinstance(obs, tuple):
            return obs, {}
        return obs

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        else:
            return result

def make_env(seed):
    """Create and return a GymAPIWrapper-wrapped Crafter environment."""
    try:
        env_base = crafter.Env()
        if hasattr(env_base, 'action_space'):
            env = GymAPIWrapper(env_base, seed=seed)
        else:
            env_registered = gym.make("CrafterReward-v1")
            while hasattr(env_registered, 'env') and hasattr(env_registered, '_max_episode_steps'):
                env_registered = env_registered.env
            env = GymAPIWrapper(env_registered, seed=seed)
    except Exception as e:
        print(f"Error with Crafter environment setup: {e}")
        env_fallback = gym.make("CrafterReward-v1")
        env = GymAPIWrapper(env_fallback, seed=seed)
    return env