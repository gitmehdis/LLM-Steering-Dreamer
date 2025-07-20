"""
Baseline episode inspector for detailed analysis of agent behavior.
"""

import numpy as np
import jax.numpy as jnp
from collections import defaultdict
import json


def inspect_baseline_episode(agent, obs_space, act_space, env, action_id_to_name,
                           inspection_seed=888, max_steps=2000, verbose=True):
    """
    Run a baseline episode with detailed logging for behavioral analysis.
    
    Args:
        agent: DreamerV3 agent instance
        obs_space: Observation space specification
        act_space: Action space specification
        env: Crafter environment instance
        action_id_to_name: Dictionary mapping action IDs to names
        inspection_seed: Random seed for reproducible episodes
        max_steps: Maximum episode length
        verbose: Whether to print detailed progress
        
    Returns:
        dict: Comprehensive inspection data with logs, statistics, and behavioral analysis
    """
    
    print("=" * 80)
    print(f"BASELINE EPISODE INSPECTION (Seed: {inspection_seed})")
    print("=" * 80)
    
    # Set seed and reset environment
    np.random.seed(inspection_seed)
    
    if hasattr(env, 'seed'):
        env.seed(inspection_seed)
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, info = obs
    else:
        info = {}
    
    # Initialize observation wrapper
    raw_obs = {
        'image': obs,
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }
    
    # Initialize agent and tracking
    carry = agent.init_policy(batch_size=1)
    tracking_data = _initialize_tracking()
    
    print(f"Starting baseline episode with seed {inspection_seed}")
    print(f"Initial observation shape: {obs.shape}")
    print()
    
    # Main episode loop
    for step in range(max_steps):
        # Process observation for agent
        obs_proc = _process_observation(obs_space, raw_obs)
        
        # Get agent action
        carry, act, extra = agent.policy(carry, obs_proc, mode='eval')
        act_np = {k: np.asarray(v[0]) for k, v in act.items()}
        action = int(act_np['action'])
        action_name = action_id_to_name.get(action, f"Unknown({action})")
        
        # Track action frequency
        tracking_data['action_counts'][action_name] += 1
        
        # Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs2, reward, term, trunc, info = step_result
        else:
            obs2, reward, done, info = step_result
            term, trunc = done, False
        
        # Update tracking data
        _update_tracking(tracking_data, step, action, action_name, reward, info)
        
        # Log step details
        step_log = _create_step_log(step, action, action_name, reward, 
                                  tracking_data['total_reward'], info, term, trunc)
        tracking_data['detailed_log'].append(step_log)
        tracking_data['action_sequence'].append(action_name)
        
        # Verbose logging for key events
        if verbose:
            _verbose_logging(step, action_name, reward, step_log, tracking_data['total_reward'])
        
        # Update for next iteration
        raw_obs = {
            'image': obs2,
            'reward': reward,
            'is_first': False,
            'is_last': term or trunc,
            'is_terminal': term or trunc,
        }
        obs = obs2
        
        if term or trunc:
            break
    
    # Generate final inspection data
    inspection_data = _generate_final_data(tracking_data, info, inspection_seed)
    
    # Print summary
    _print_summary(inspection_data, verbose)
    
    return inspection_data


def _initialize_tracking():
    """Initialize tracking data structures."""
    return {
        'total_reward': 0.0,
        'step_count': 0,
        'achievement_unlocks': [],
        'action_counts': defaultdict(int),
        'detailed_log': [],
        'action_sequence': [],
        'previous_achievements': {}
    }


def _process_observation(obs_space, raw_obs):
    """Process raw observation for agent consumption."""
    obs_proc = {}
    for k in obs_space.keys():
        v = raw_obs.get(k, 0)
        if k == 'image':
            arr = jnp.expand_dims(jnp.array(v, dtype=jnp.uint8), 0)
        else:
            arr = jnp.expand_dims(jnp.array(v), 0)
        obs_proc[k] = arr
    return obs_proc


def _update_tracking(tracking_data, step, action, action_name, reward, info):
    """Update tracking data with current step information."""
    tracking_data['total_reward'] += reward
    tracking_data['step_count'] += 1
    
    # Detect new achievements
    current_achievements = info.get('achievements', {})
    if current_achievements:
        for ach, count in current_achievements.items():
            prev_count = tracking_data['previous_achievements'].get(ach, 0)
            if count > prev_count:
                tracking_data['achievement_unlocks'].append({
                    'step': step,
                    'achievement': ach,
                    'count': count,
                    'total_reward': tracking_data['total_reward']
                })
        tracking_data['previous_achievements'] = current_achievements.copy()


def _create_step_log(step, action, action_name, reward, cumulative_reward, info, term, trunc):
    """Create detailed log entry for current step."""
    current_inventory = info.get('inventory', {})
    current_achievements = info.get('achievements', {})
    
    return {
        'step': step,
        'action': action,
        'action_name': action_name,
        'reward': reward,
        'cumulative_reward': cumulative_reward,
        'inventory': current_inventory.copy(),
        'inventory_count': sum(current_inventory.values()) if current_inventory else 0,
        'achievements': current_achievements.copy(),
        'new_achievements': [],
        'facing': info.get('neighbor', 'unknown'),
        'done': term or trunc
    }


def _verbose_logging(step, action_name, reward, step_log, total_reward):
    """Print verbose logging for key events."""
    if reward > 0 or step_log['new_achievements'] or step % 200 == 0:
        status = ""
        if reward > 0:
            status += f"+{reward:.2f} reward "
        if step_log['new_achievements']:
            status += f"{', '.join(step_log['new_achievements'])} "
        
        print(f"Step {step:4d}: {action_name:16s} | {status} | "
              f"Inventory: {step_log['inventory_count']} items | "
              f"Total reward: {total_reward:.2f}")


def _generate_final_data(tracking_data, final_info, inspection_seed):
    """Generate comprehensive final inspection data."""
    final_achievements = final_info.get('achievements', {})
    total_achievement_count = sum(final_achievements.values()) if final_achievements else 0
    
    # Calculate behavioral statistics
    behavioral_stats = _calculate_behavioral_stats(tracking_data['action_counts'], 
                                                 tracking_data['step_count'],
                                                 tracking_data['total_reward'])
    
    return {
        'seed': inspection_seed,
        'total_steps': tracking_data['step_count'],
        'total_reward': tracking_data['total_reward'],
        'final_achievements': final_achievements,
        'achievement_count': total_achievement_count,
        'achievement_unlocks': tracking_data['achievement_unlocks'],
        'action_counts': dict(tracking_data['action_counts']),
        'action_sequence': tracking_data['action_sequence'],
        'detailed_log': tracking_data['detailed_log'],
        'final_inventory': final_info.get('inventory', {}),
        'behavioral_stats': behavioral_stats
    }


def _calculate_behavioral_stats(action_counts, step_count, total_reward):
    """Calculate behavioral pattern statistics."""
    if step_count == 0:
        return {}
    
    # Movement analysis
    movement_actions = ['Move Left', 'Move Right', 'Move Up', 'Move Down']
    total_movement = sum(action_counts.get(action, 0) for action in movement_actions)
    movement_ratio = total_movement / step_count
    
    # Interaction analysis
    interact_count = action_counts.get('Do (Interact)', 0)
    interact_ratio = interact_count / step_count
    
    # Crafting analysis
    crafting_actions = ['Make Wood Pickaxe', 'Make Stone Pickaxe', 'Make Iron Pickaxe', 
                       'Make Wood Sword', 'Make Stone Sword', 'Make Iron Sword']
    total_crafting = sum(action_counts.get(action, 0) for action in crafting_actions)
    
    # Placing analysis
    placing_actions = ['Place Stone', 'Place Table', 'Place Furnace', 'Place Plant']
    total_placing = sum(action_counts.get(action, 0) for action in placing_actions)
    
    return {
        'movement_ratio': movement_ratio,
        'interact_ratio': interact_ratio,
        'total_crafting': total_crafting,
        'total_placing': total_placing,
        'reward_per_step': total_reward / step_count if step_count > 0 else 0
    }


def _print_summary(inspection_data, verbose=True):
    """Print comprehensive summary of inspection results."""
    print()
    print("=" * 80)
    print("EPISODE INSPECTION SUMMARY")
    print("=" * 80)
    print(f"Seed: {inspection_data['seed']}")
    print(f"Total steps: {inspection_data['total_steps']}")
    print(f"Total reward: {inspection_data['total_reward']:.2f}")
    print(f"Final achievements: {inspection_data['achievement_count']}")
    print(f"Achievement unlocks: {len(inspection_data['achievement_unlocks'])}")
    print()
    
    if not verbose:
        return
    
    # Achievement timeline
    if inspection_data['achievement_unlocks']:
        print("ACHIEVEMENT TIMELINE:")
        print("-" * 50)
        for unlock in inspection_data['achievement_unlocks']:
            print(f"  Step {unlock['step']:4d}: {unlock['achievement']} "
                  f"(reward: {unlock['total_reward']:.2f})")
        print()
    
    # Action frequency analysis
    print("ACTION FREQUENCY ANALYSIS:")
    print("-" * 50)
    sorted_actions = sorted(inspection_data['action_counts'].items(), 
                          key=lambda x: x[1], reverse=True)
    for action_name, count in sorted_actions:
        percentage = (count / inspection_data['total_steps']) * 100
        print(f"  {action_name:16s}: {count:4d} times ({percentage:5.1f}%)")
    print()
    
    # Resource collection summary
    if inspection_data['final_achievements']:
        print("FINAL INVENTORY & ACHIEVEMENTS:")
        print("-" * 50)
        if inspection_data['final_inventory']:
            print("Inventory:")
            for item, count in sorted(inspection_data['final_inventory'].items()):
                if count > 0:
                    print(f"  {item}: {count}")
        
        print("Achievements:")
        for ach, count in sorted(inspection_data['final_achievements'].items()):
            if count > 0:
                print(f"  {ach}: {count}")
        print()
    
    # Behavioral patterns
    print("BEHAVIORAL PATTERNS:")
    print("-" * 50)
    stats = inspection_data['behavioral_stats']
    print(f"Movement ratio: {stats.get('movement_ratio', 0):.1%} of actions")
    print(f"Interaction ratio: {stats.get('interact_ratio', 0):.1%} of actions")
    print(f"Crafting actions: {stats.get('total_crafting', 0)} total")
    print(f"Placing actions: {stats.get('total_placing', 0)} total")
    print(f"Reward efficiency: {stats.get('reward_per_step', 0):.4f} reward per step")
    print()


def save_inspection_log(inspection_data, filename=None):
    """Save inspection data to JSON file."""
    if filename is None:
        filename = f'baseline_inspection_seed_{inspection_data["seed"]}.json'
    
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    for key, value in inspection_data.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = value.item()
        else:
            serializable_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Detailed log saved to: {filename}")
    return filename
