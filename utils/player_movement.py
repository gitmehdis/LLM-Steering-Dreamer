import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

def extract_player_fov_semantic(semantic_map, player_pos, player_facing, view_rows=18, view_cols=18, item_rows=4):
    """
    Extract and align the player's field of view from the full semantic map.
    
    Applies the correct coordinate transformation to match the visual RGB image:
    1. Extract FOV crop using correct (row,col) coordinates
    2. Apply rotation and flip transformations for visual alignment
    
    Args:
        semantic_map: Full world semantic map (64x64)
        player_pos: Player position as (row, col) = (y, x)
        player_facing: Player facing direction (0=North, 1=East, 2=South, 3=West)
        view_rows, view_cols: FOV dimensions (default 18x18)
        item_rows: Columns reserved for inventory (default 4)
    
    Returns:
        aligned_semantic: Properly oriented FOV semantic map
        player_fov_pos: Player position in aligned FOV (should be center)
        fov_bounds: Original crop bounds in world coordinates
    """
    
    world_fov_cols = view_cols - item_rows  # 18 - 4 = 14
    fov_rows_radius = view_rows // 2        # 9 tiles each direction
    fov_cols_radius = world_fov_cols // 2   # 7 tiles each direction
    
    # Correct coordinate unpacking: player_pos is (row, col) = (y, x)
    py, px = int(player_pos[0]), int(player_pos[1])
    
    # Extract FOV crop
    y0 = max(0, py - fov_rows_radius)
    y1 = min(semantic_map.shape[0], py + fov_rows_radius + 1)
    x0 = max(0, px - fov_cols_radius) 
    x1 = min(semantic_map.shape[1], px + fov_cols_radius + 1)
    
    raw_crop = semantic_map[y0:y1, x0:x1]
    
    # Apply transformation sequence to align with visual image
    # This combines: facing-based rotation + horizontal flip + final 90° CCW rotation
    facing_rotation = [0, 1, 2, 3][player_facing % 4]  # Rotation based on facing direction
    
    # Complete transformation pipeline
    rotated = np.rot90(raw_crop, k=facing_rotation)      # Orient forward=up
    flipped = np.fliplr(rotated)                         # Match RGB left/right  
    aligned_semantic = np.rot90(flipped, k=1)            # Final alignment
    
    # Player position in aligned map (should be center)
    player_fov_pos = (aligned_semantic.shape[0] // 2, aligned_semantic.shape[1] // 2)
    
    return aligned_semantic, player_fov_pos, (y0, x0, y1, x1)

def analyze_semantic_map_fov(fov_semantic, mat_ids, obj_ids, player_fov_pos):
    """Analyze only the player's field of view semantic map"""
    
    reverse_mat_ids = {v: k for k, v in mat_ids.items()}
    reverse_obj_ids = {v: k for k, v in obj_ids.items()}
    
    # Get unique values and their counts in FOV
    unique_values, counts = np.unique(fov_semantic, return_counts=True)
    
    # FOV counts
    fov_counts = {}
    for value, count in zip(unique_values, counts):
        if value in reverse_mat_ids:
            item_name = str(reverse_mat_ids[value])
        elif value in reverse_obj_ids:
            item_name = str(reverse_obj_ids[value])
        else:
            item_name = f'unknown_{value}'
        fov_counts[item_name] = count
    
    return fov_counts
def move_player_by_path(env, agent, mat_ids, obj_ids, path_directions, current_obs, current_info, extract_player_fov_semantic, analyze_semantic_map_fov, verify_pov=True, show_images=False):
    """
    Move the player in the world according to a list of directions and output the new semantic POV for verification.
    
    Args:
        env: The game environment
        agent: The DreamerV3 agent
        mat_ids: Material ID mapping
        obj_ids: Object ID mapping
        path_directions: List of direction strings ('up', 'down', 'left', 'right')
        current_obs: Current observation from environment
        current_info: Current info dict from environment
        extract_player_fov_semantic: Function to extract FOV
        analyze_semantic_map_fov: Function to analyze FOV
        verify_pov: If True, outputs the new semantic POV after movement
        show_images: If True, show game images after each movement step
    
    Returns:
        final_obs: Final observation after all movements
        final_semantic_pov: Final player's POV semantic map
        movement_log: List of (step, direction, position, action_taken, facing) for each movement
    """
    if not path_directions:
        print("No path directions provided")
        return None, None, []
    
    # Direction to action mapping
    direction_to_action = {
        'up': 3,     # Move up
        'down': 4,   # Move down  
        'left': 1,   # Move left
        'right': 2   # Move right
    }
    
    # Direction mapping for manual tracking
    direction_to_facing = {
        'up': 0,     # North
        'right': 1,  # East
        'down': 2,   # South
        'left': 3    # West
    }
    
    # Facing names for display
    facing_names = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
    
    print(f"Starting movement sequence: {path_directions}")
    
    # Track movements
    movement_log = []
    carry = agent.init_policy(batch_size=1)
    
    # Use the passed current state (not global variables)
    obs = current_obs
    info = current_info
    initial_pos = info.get('player_pos', np.array([32, 32]))
    
    # Manual facing direction tracking
    current_facing = 0  # Start facing North (arbitrary default)
    
    print(f"Starting position: {tuple(initial_pos)}, facing: {current_facing} ({facing_names[current_facing]})")
    
    for step, direction in enumerate(path_directions):
        if direction not in direction_to_action:
            print(f"Invalid direction '{direction}' at step {step}. Skipping.")
            continue
            
        action = direction_to_action[direction]
        current_facing = direction_to_facing[direction]
        
        print(f"Step {step + 1}: Moving {direction} (action {action}), now facing: {current_facing} ({facing_names[current_facing]})")
        
        # Execute the movement action
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, term, trunc, info = step_result
        else:
            obs, reward, done, info = step_result
            term, trunc = done, False
        
        # Show game image after movement if requested
        if show_images:
            obs_uint8 = obs.astype(np.uint8)
            plt.figure(figsize=(6, 6))
            plt.imshow(obs_uint8)
            plt.title(f"After Movement Step {step + 1}: {direction.capitalize()}", fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Log the movement
        new_pos = info.get('player_pos', initial_pos)
        movement_log.append((step + 1, direction, tuple(new_pos), action, current_facing))
        
        print(f"  New position: {tuple(new_pos)}, facing: {current_facing} ({facing_names[current_facing]})")
        
        if term or trunc:
            print(f"Environment terminated at step {step + 1}")
            break
    
    # Get final state and POV
    final_obs = obs
    final_info = info
    final_pos = final_info.get('player_pos', initial_pos)
    final_semantic_map = final_info.get('semantic', np.zeros((64, 64), dtype=np.int32))
    
    print(f"Movement completed. Final position: {tuple(final_pos)}, final facing: {current_facing} ({facing_names[current_facing]})")
    
    # Extract and display final POV if requested
    final_semantic_pov = None
    if verify_pov:
        print(f"\nExtracting final semantic POV for verification...")
        consistent_facing = 0  # Always use North for consistent comparison
        fov_semantic, player_fov_pos, fov_bounds = extract_player_fov_semantic(
            final_semantic_map, final_pos, consistent_facing)
        
        final_semantic_pov = fov_semantic
        
        print(f"Final POV shape: {fov_semantic.shape}")
        print(f"Player position in POV: {player_fov_pos}")
        print(f"POV bounds in world: rows {fov_bounds[0]}-{fov_bounds[2]}, cols {fov_bounds[1]}-{fov_bounds[3]}")
        print(f"Player actual facing: {current_facing} ({facing_names[current_facing]})")
        print(f"POV extracted using consistent facing: {consistent_facing} (North) for comparison")
        print(f"\nFinal Semantic POV:")
        print(fov_semantic)
        
        # Analyze the POV contents
        fov_counts = analyze_semantic_map_fov(fov_semantic, mat_ids, obj_ids, player_fov_pos)
        print(f"\nItems visible in final POV:")
        for item_name, count in sorted(fov_counts.items(), key=lambda x: str(x[0])):
            print(f"   - {str(item_name)}: {count} tiles")
    
    return final_obs, final_semantic_pov, movement_log

def process_llm_moves(move_sequence, task, env, agent, obs, info, show_images=True):
    """
    Process a sequence of moves from the LLM for 'tunnel' or 'bridge' tasks.
    For 'tunnel': collect stone at each step.
    For 'bridge': place stone, collect it (to make a path), and move there.
    
    Args:
        move_sequence: List of move directions from LLM
        task: Either 'tunnel' or 'bridge'
        env: Game environment
        agent: DreamerV3 agent
        obs: Current observation
        info: Current info dict
        show_images: Whether to show images after each construction step
    """
    direction_to_action = {
        'up': 3,     # Move up
        'down': 4,   # Move down  
        'left': 1,   # Move left
        'right': 2   # Move right
    }
    
    collect_action = 5  # "Do (Interact)" - for collecting/mining stone
    place_action = 7    # "Place Stone" - for placing stone

    carry = agent.init_policy(batch_size=1)
    movement_log = []
    
    # Show initial state before construction
    if show_images:
        obs_uint8 = obs.astype(np.uint8)
        plt.figure(figsize=(6, 6))
        plt.imshow(obs_uint8)
        plt.title(f"Initial State - Before {task.capitalize()} Construction", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    for i, move in enumerate(move_sequence):
        move = move.strip().lower()
        if move not in direction_to_action:
            print(f"Invalid move '{move}' in LLM sequence.")
            continue
            
        print(f"Construction Step {i+1}/{len(move_sequence)}: {move}")
        
        if task == "tunnel":
            # Collect stone at current location
            step_result = env.step(collect_action)
            if len(step_result) == 5:
                obs, reward, term, trunc, info = step_result
            else:
                obs, reward, done, info = step_result
                term, trunc = done, False
            print(f"  Collected stone at {info.get('player_pos')}")
            
            # Show intermediate state after collection
            if show_images:
                obs_uint8 = obs.astype(np.uint8)
                plt.figure(figsize=(6, 6))
                plt.imshow(obs_uint8)
                plt.title(f"After Stone Collection - Step {i+1}", fontsize=12)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # Move in the specified direction
            step_result = env.step(direction_to_action[move])
            if len(step_result) == 5:
                obs, reward, term, trunc, info = step_result
            else:
                obs, reward, done, info = step_result
                term, trunc = done, False
            print(f"  Moved {move} to {info.get('player_pos')}")
            
        elif task == "bridge":
            # Place stone at current location
            step_result = env.step(place_action)
            if len(step_result) == 5:
                obs, reward, term, trunc, info = step_result
            else:
                obs, reward, done, info = step_result
                term, trunc = done, False
            print(f"  Placed stone at {info.get('player_pos')}")
            
            # Show intermediate state after placing
            if show_images:
                obs_uint8 = obs.astype(np.uint8)
                plt.figure(figsize=(6, 6))
                plt.imshow(obs_uint8)
                plt.title(f"After Stone Placement - Step {i+1}", fontsize=12)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # Collect stone to make it a path
            step_result = env.step(collect_action)
            if len(step_result) == 5:
                obs, reward, term, trunc, info = step_result
            else:
                obs, reward, done, info = step_result
                term, trunc = done, False
            print(f"  Made path at {info.get('player_pos')}")
            
            # Move in the specified direction
            step_result = env.step(direction_to_action[move])
            if len(step_result) == 5:
                obs, reward, term, trunc, info = step_result
            else:
                obs, reward, done, info = step_result
                term, trunc = done, False
            print(f"  Moved {move} to {info.get('player_pos')}")
            
        else:
            print(f"Unknown task '{task}'")
            continue
            
        # Show final state after complete move
        if show_images:
            obs_uint8 = obs.astype(np.uint8)
            plt.figure(figsize=(6, 6))
            plt.imshow(obs_uint8)
            plt.title(f"After Move {i+1}: {move.capitalize()}", fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        movement_log.append((move, info.get('player_pos')))
        
        if term or trunc:
            print("Environment terminated during LLM move sequence.")
            break
            
    print(f"Construction completed: {len(movement_log)} moves executed")
    return obs, info, movement_log

