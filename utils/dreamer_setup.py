import os
from typing import Dict
import jax
try:
    from jax.experimental.compilation_cache import set_cache_dir
except ImportError:
    def set_cache_dir(path):
        print("[WARN] JAX set_cache_dir not available in this JAX version. Skipping persistent cache setup.")
import ruamel.yaml as yaml
from dreamerv3.agent import Agent
import elements
from embodied import wrappers

# ---------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------
ACTION_ID_TO_NAME: Dict[int, str] = {
    0: "Noop", 1: "Move Left", 2: "Move Right", 3: "Move Up", 4: "Move Down",
    5: "Do (Interact)", 6: "Sleep", 7: "Place Stone", 8: "Place Table",
    9: "Place Furnace", 10: "Place Plant", 11: "Make Wood Pickaxe",
    12: "Make Stone Pickaxe", 13: "Make Iron Pickaxe", 14: "Make Wood Sword",
    15: "Make Stone Sword", 16: "Make Iron Sword",
}

# ---------------------------------------------------------------------
# 1) JAX COMPILE CACHE
# ---------------------------------------------------------------------
set_cache_dir("/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update('jax_transfer_guard', 'allow')

# ---------------------------------------------------------------------
# DREAMERV3 AGENT LOADING (from agent_mod.py)
# ---------------------------------------------------------------------
def load_dreamer_agent(logdir, ckpt_rel, seed):
    config_path = os.path.join(logdir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.YAML(typ='safe').load(f)
    config['seed'] = seed
    config['logdir'] = logdir
    config['script'] = 'eval_only'
    config['task'] = 'crafter_reward'
    if 'jax' not in config:
        config['jax'] = {}
    config['jax']['platform'] = 'cpu' # Use CPU for evaluation
    config['jax']['expect_devices'] = 1
    config['jax']['transfer_guard'] = False
    # Setup environment for obs/act space
    from embodied.envs.crafter import Crafter
    env_cfg = config['env']['crafter']
    env = Crafter('reward', **env_cfg)
    env = wrappers.UnifyDtypes(env)
    env = wrappers.CheckSpaces(env)
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    # Load agent
    flat_cfg = dict(config)
    flat_cfg.update(config.get('agent', {}))
    flat_cfg['jax'] = config.get('jax', {})
    agent = Agent(obs_space, act_space, elements.Config(**flat_cfg))
    cp = elements.Checkpoint()
    cp.agent = agent
    ckpt_path = os.path.abspath(os.path.join(logdir, ckpt_rel))
    cp.load(ckpt_path, keys=['agent'])
    return agent, obs_space, act_space, env

