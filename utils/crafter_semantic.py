from crafter import Env
import pprint

def inspect_crafter_semantics(verbose=True):
    env = Env()
    obs = env.reset()

    # Take one action so we get an info dict
    obs, reward, done, info = env.step(env.action_space.sample())

    if verbose:
        # 1. Print all the keys & values in info
        print("INFO KEYS:", info.keys())
        pprint.pprint(info)                  # pretty-print the entire dict

    # 2. Look at the semantic map itself
    semantic_map = info['semantic']
    if verbose:
        print("SEMANTIC MAP shape:", semantic_map.shape)
        print(semantic_map)                  # prints the 2D array of IDs

    # 3. Decode those integer IDs back to names
    mat_ids = env._sem_view._mat_ids     # mapping material_name → integer
    obj_ids = env._sem_view._obj_ids     # mapping object_class → integer

    if verbose:
        print("\nMATERIAL ID → name:")
        for name, mid in mat_ids.items():
            print(f"  {mid:3d} → {name}")

        print("\nOBJECT ID → class:")
        for cls, oid in obj_ids.items():
            print(f"  {oid:3d} → {cls.__name__}")

    # Take another step to get updated info
    obs, rew, done, info = env.step(env.action_space.sample())
    semantic_map = info['semantic']          # (64×64) integer array including the player ID
    player_pos   = info['player_pos']       # e.g. array([x, y])
    facing       = tuple(env._player.facing)  # e.g. (0, 1)
    if verbose:
        print("At", player_pos, "player is facing", facing)

    # Return useful mappings for later use
    return mat_ids, obj_ids, semantic_map, player_pos, facing