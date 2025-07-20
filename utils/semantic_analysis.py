"""
Semantic map analysis and field-of-view extraction utilities for Crafter.
"""
import numpy as np


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
