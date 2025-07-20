import openai
import matplotlib.pyplot as plt
import numpy as np
import os
import jax.numpy as jnp

def LLM_steering(env, agent, mat_ids, obj_ids, crafter_colormap, extract_player_fov_semantic, analyze_semantic_map_fov, move_player_by_path, process_llm_moves, PathFinding, steps_to_run=100, initial_steps=20, cooldown_steps=30, input_seed=888, show_navigation_images=False, show_construction_images=True):
    """Run steps with conditional LLM calls based on inventory and POV contents"""
    
    # Get OpenAI API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    client = openai.OpenAI(api_key=openai_api_key)
    

    
    # Reset environment and run some steps
    np.random.seed(input_seed)
    img0 = env.reset()
    if isinstance(img0, tuple):
        obs, info = img0
    else:
        obs = img0
        info = {}
    
    # LLM calling state
    last_llm_step = -cooldown_steps  # Allow first call after initial_steps
    llm_call_count = 0
    
    # Run a few steps
    carry = agent.init_policy(batch_size=1)
    print(f"Running {steps_to_run} steps with LLM conditions...")
    print(f"LLM will be called after step {initial_steps} if conditions are met")
    print(f"Cooldown period: {cooldown_steps} steps between LLM calls")
    
    def check_inventory_condition(inventory):
        """Check if inventory has stone or wood pickaxe"""
        wood_pickaxe = inventory.get('wood_pickaxe', 0) > 0
        stone_pickaxe = inventory.get('stone_pickaxe', 0) > 0
        return wood_pickaxe or stone_pickaxe
    
    def check_pov_condition(fov_semantic):
        """Check if stone (ID=3) or water (ID=1) is in POV"""
        unique_ids = set(np.unique(fov_semantic))
        return 1 in unique_ids or 3 in unique_ids  # Water or Stone
    
    def should_call_llm(step, inventory, fov_semantic):
        """Determine if LLM should be called this step"""
        nonlocal last_llm_step
        
        # Check timing conditions
        initial_steps_passed = step >= initial_steps
        cooldown_complete = (step - last_llm_step) >= cooldown_steps
        
        if not (initial_steps_passed and cooldown_complete):
            return False
        
        # Check game state conditions
        has_pickaxe = check_inventory_condition(inventory)
        has_target_in_pov = check_pov_condition(fov_semantic)
        
        return has_pickaxe and has_target_in_pov
    
    for step in range(steps_to_run):
        # Prepare observation
        obs_proc = {
            'image': jnp.expand_dims(jnp.array(obs, dtype=jnp.uint8), 0),
            'reward': jnp.expand_dims(jnp.array(0.0), 0),
            'is_first': jnp.expand_dims(jnp.array(step == 0), 0),
            'is_last': jnp.expand_dims(jnp.array(False), 0),
            'is_terminal': jnp.expand_dims(jnp.array(False), 0),
        }
        
        # Get agent action
        carry, act, extra = agent.policy(carry, obs_proc, mode='eval')
        act_np = {k: np.asarray(v[0]) for k, v in act.items()}
        action = int(act_np['action'])
        
        # Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, term, trunc, info = step_result
        else:
            obs, reward, done, info = step_result
            term, trunc = done, False
        
        # Check LLM calling conditions
        current_inventory = info.get('inventory', {})
        current_semantic = info.get('semantic', np.zeros((64, 64), dtype=np.int32))
        current_player_pos = info.get('player_pos', np.array([32, 32]))
        
        # Extract current FOV for condition checking
        current_fov, _, _ = extract_player_fov_semantic(current_semantic, current_player_pos, 0, view_rows=18, view_cols=18, item_rows=4)
        
        if should_call_llm(step, current_inventory, current_fov):
            print(f"\n Step {step}: LLM conditions met!")
            print(f"   Inventory: {current_inventory}")
            print(f"   Has pickaxe: {check_inventory_condition(current_inventory)}")
            print(f"   Stone/Water in POV: {check_pov_condition(current_fov)}")

            # Update LLM call tracking
            last_llm_step = step
            llm_call_count += 1
            
            # --- LLM call and action logic moved here ---
            # Extract synchronized data after LLM conditions are met
            full_semantic_map = info.get('semantic', np.zeros((64, 64), dtype=np.int32))
            player_pos = info.get('player_pos', np.array([32, 32]))
            inventory = info.get('inventory', {})
            
            print(f"\n Current state at step {step + 1}:")
            print(f" Obs shape: {obs.shape}, Full semantic shape: {full_semantic_map.shape}")
            print(f" Player at: {tuple(player_pos)}")
            print(f" Inventory: {inventory}")
            print(f" LLM calls made: {llm_call_count}")
            
            # Only proceed with LLM if conditions were met
            # (This check is not needed anymore, since we're inside the condition)
            
            # Extract player's field of view (18x14 tiles) with proper alignment
            player_dir = 0  # Always use North for consistent POV orientation
            fov_semantic, player_fov_pos, fov_bounds = extract_player_fov_semantic(
                full_semantic_map, player_pos, player_dir, view_rows=36, view_cols=36, item_rows=8)
            print(f" Player FOV shape: {fov_semantic.shape}, Player FOV pos: {player_fov_pos}")
            print(f" FOV bounds in world: rows {fov_bounds[0]}-{fov_bounds[2]}, cols {fov_bounds[1]}-{fov_bounds[3]}")
            
            # Analyze only the FOV semantic map
            fov_counts = analyze_semantic_map_fov(fov_semantic, mat_ids, obj_ids, player_fov_pos)
            
            # Create semantic information text focused on FOV
            semantic_text = f"PLAYER FIELD OF VIEW ANALYSIS:\n"
            semantic_text += f"- FOV size: {fov_semantic.shape[0]}x{fov_semantic.shape[1]} tiles (18 rows × 14 cols world view)\n"
            semantic_text += f"- Player position in FOV: {player_fov_pos}\n"
            semantic_text += f"- World coordinates: rows {fov_bounds[0]}-{fov_bounds[2]}, cols {fov_bounds[1]}-{fov_bounds[3]}\n\n"
            
            semantic_text += "ITEMS VISIBLE IN PLAYER'S FOV:\n"
            for item_name, count in sorted(fov_counts.items(), key=lambda x: str(x[0])):
                semantic_text += f"- {str(item_name)}: {count} tiles visible\n"
            
            semantic_text += f"\nSPATIAL LAYOUT IN FOV (relative to player):\n"
            semantic_text += f"Semantic map values:\n{fov_semantic}\n\n"
            
            # Display initial state image
            print("\n INITIAL STATE - Before LLM object selection:")
            obs_uint8 = obs.astype(np.uint8)
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(obs_uint8)
            plt.title("Initial Visual Image", fontsize=14)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(fov_semantic, cmap=crafter_colormap, vmin=0, vmax=19)
            plt.title("Initial FOV Semantic Map", fontsize=14)
            plt.colorbar(label='Material/Object ID', shrink=0.8)
            plt.plot(player_fov_pos[1], player_fov_pos[0], 'r*', markersize=20,
                    markeredgecolor='white', markeredgewidth=2, label='Player')
        
            
            plt.tight_layout()
            plt.show()
            
            # Ask LLM to select an object
            print(f"\n Asking LLM to select an object from the scene...")
            
            # Create object mapping for LLM
            reverse_mat_ids = {v: k for k, v in mat_ids.items()}
            reverse_obj_ids = {v: k for k, v in obj_ids.items()}
            
            # Get available objects (non-walkable items in FOV)
            walkable_ids = {2, 4, 5}  # Based on pathfinding walkable set
            unique_ids = set(np.unique(fov_semantic)) - walkable_ids
            available_objects = []
            
            for obj_id in unique_ids:
                if obj_id in reverse_mat_ids:
                    obj_name = reverse_mat_ids[obj_id]
                elif obj_id in reverse_obj_ids:
                    obj_name = reverse_obj_ids[obj_id]
                else:
                    obj_name = f'unknown_{obj_id}'
                available_objects.append((obj_id, obj_name))
            
            if not available_objects:
                print(" No non-walkable objects found in FOV")
                return fov_semantic, player_fov_pos
            
            # Prepare LLM prompt
            objects_list = "\n".join([f"- ID {obj_id}: {obj_name}" for obj_id, obj_name in available_objects])
            
            llm_prompt = f"""You are helping a player navigate in a Minecraft-like world. Based on the semantic analysis below, please select ONE object that would be interesting or useful for the player to move towards.

{semantic_text}

Available objects in the player's field of view:

{objects_list}

In the semantic map, numbers (1-18) represent different materials or objects:
Material IDs: 'water': 1, 'grass': 2, 'stone': 3, 'path': 4, 'sand': 5, 'tree': 6, 'lava': 7, 'coal': 8, 'iron': 9, 'diamond': 10, 'table': 11, 'furnace': 12
Object IDs: 'Player': 13, 'Cow': 14, 'Zombie': 15, 'Skeleton': 16, 'Arrow': 17, 'Plant': 18

Please write a sentence describing the pov of the player, matching the image and the semnatic map. 

Then please respond with whether you want to go to either water or stone to build either build a bridge on water or tunnel in stone. 

Building a bridge requires stone in the inventory while tunneling requires a pickaxe. Here is your inventory:
{current_inventory}

Your response MUST be of the format: The first like is for the pov description and the next line is ONLY a number which is 1 for water or 3 for stone for your choice.

"""

            try:
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that helps a player navigate and build interesting structures in the Minecraft-style game of Crafter."},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.3
                )
                
                # extract LLM response and print the first line for pov description
                print(f" LLM response:\n{response.choices[0].message.content}")
                # Parse LLM response
                # Expecting response format: "pov description\nselected_object_id"
                lines = [line.strip() for line in response.choices[0].message.content.strip().split('\n') if line.strip()]
                if len(lines) < 2:
                    print(" LLM response does not contain enough lines. Expected at least 2 lines.")
                    return fov_semantic, player_fov_pos
                # The first non-empty line is the description, next valid int line is the object ID
                print(f" LLM description: {lines[0]}")
                selected_object_id = None
                for line in lines[1:]:
                    try:
                        val = int(line)
                        if val in [1, 3]:
                            selected_object_id = val
                            print(f" LLM selected object ID: {selected_object_id}")
                            break
                    except Exception:
                        continue
                if selected_object_id is None:
                    print(f" No valid object ID (1 or 3) found in LLM response lines: {lines}")
                    return fov_semantic, player_fov_pos
                # Map LLM ID to object name
                selected_object_name = None
                for obj_id, obj_name in available_objects:
                    if obj_id == selected_object_id:
                        selected_object_name = obj_name
                        break
                if selected_object_name is None:
                    print(f" Invalid object ID selected by LLM: {selected_object_id}. No matching object found.")
                    return fov_semantic, player_fov_pos

            except Exception as e:
                print(f" OpenAI API Error: {e}")
                # Fallback to first available object
                selected_object_id = available_objects[0][0]
                selected_object_name = available_objects[0][1]
                print(f" Fallback: Using object ID {selected_object_id} ({selected_object_name})")
            
            # Use pathfinding to find path to the selected object
            print(f"\n Finding path to {selected_object_name} (ID {selected_object_id})...")
            
            pf = PathFinding(walkable={2, 4, 5}, grid_size=fov_semantic.shape)
            path_directions = pf.find(fov_semantic, start=player_fov_pos, target_item=selected_object_id)
            
            if not path_directions:
                print(f" No path found to {selected_object_name}")
                continue  # Instead of return, continue to next step

            print(f" Path found: {path_directions}")

            # Move player along the path
            print(f"\n Moving player to {selected_object_name}...")
            final_obs, final_semantic_pov, movement_log = move_player_by_path(
                env=env,
                agent=agent,
                mat_ids=mat_ids,
                obj_ids=obj_ids,
                path_directions=path_directions,
                current_obs=obs,
                current_info=info,
                extract_player_fov_semantic=extract_player_fov_semantic,
                analyze_semantic_map_fov=analyze_semantic_map_fov,
                verify_pov=True,
                show_images=show_navigation_images
            )
            if final_obs is None:
                print(" Movement failed")
                continue  # Instead of return, continue to next step

            # Display final state image
            print(f"\n FINAL STATE - After moving to {selected_object_name}:")
            final_obs_uint8 = final_obs.astype(np.uint8)
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(final_obs_uint8)
            plt.title(f"Final Visual Image (After {selected_object_name})", fontsize=14)
            plt.axis('off')
            
            if final_semantic_pov is not None:
                plt.subplot(1, 2, 2)
                plt.imshow(final_semantic_pov, cmap=crafter_colormap, vmin=0, vmax=19)
                plt.title("Final FOV Semantic Map", fontsize=14)
                plt.colorbar(label='Material/Object ID', shrink=0.8)
                center_pos = (final_semantic_pov.shape[0] // 2, final_semantic_pov.shape[1] // 2)
                plt.plot(center_pos[1], center_pos[0], 'r*', markersize=20,
                        markeredgecolor='white', markeredgewidth=2, label='Player')


            plt.tight_layout()
            plt.show()
            
            print(f"\n Movement Summary:")
            for step_num, direction, position, action, facing in movement_log:
                facing_name = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}[facing]
                print(f"  Step {step_num}: Moved {direction} to {position} (action {action}), facing {facing} ({facing_name})")
            
            print(f"\n Successfully moved to {selected_object_name}!")


            # --- New LLM call for move sequence after movement ---
            print("\n Asking LLM for move sequence to build bridge or tunnel at new location...")
            # Prepare new prompt
            new_semantic_text = (
                "This is the second part of the task. "
                f"In the first part, you have selected the player to move adjacent to nearest {selected_object_name}. If you selected water, you will build a bridge, if you selected stone, you will build a tunnel.\n"
                
                "After moving, here is the new field of view and semantic map.\n"
                f"Semantic map values:\n{final_semantic_pov}\n\n"
                "In the semantic map, numbers (1-18) represent different materials or objects:\n"
                "Material IDs: {'water': 1, 'grass': 2, 'stone': 3, 'path': 4, 'sand': 5, 'tree': 6, 'lava': 7, 'coal': 8, 'iron': 9, 'diamond': 10, 'table': 11, 'furnace': 12}\n"
                "Object IDs: {'Player': 13, 'Cow': 14, 'Zombie': 15, 'Skeleton': 16, 'Arrow': 17, 'Plant': 18}\n"

                "Based on your previous decision (bridge or tunnel), "
                f"please first write in a single line a description of the current scene to help decide the moves. Note how the player is currently facing {facing} ({facing_name})\n"
                "Then, on the next line, specify a sequence of moves (right, left, up, down) to build the structure. "
                "Use your analysis of the semantic map and the player's position to determine the best moves. Your moves should depend on the shape of the bridge or tunnel you are going to build.\n"
                "List moves separated by commas, e.g., up, up, right, down.\n"
                "Based on your analysis, aim for p-shape (or its rotation) bridge or tunnel when possible.\n"
                "Respond with two lines: first line is the scene description, second line is the move sequence."
            )

            llm_prompt2 = (
                f"{new_semantic_text}\n"
                f"Image and semantic map are shown above. Previously you decided to build a "
                f"{'bridge' if selected_object_id == 1 else 'tunnel'}."
            )

            try:
                response2 = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for Minecraft-style world. You help players build structures based on their decisions."},
                        {"role": "user", "content": llm_prompt2}
                    ],
                    temperature=0.3
                )
                print(f" LLM move sequence response:\n{response2.choices[0].message.content}")
                # Improved parsing: skip empty lines, first non-empty is description, next is moves
                lines2 = [line.strip() for line in response2.choices[0].message.content.strip().split('\n') if line.strip()]
                if len(lines2) < 2:
                    print("❌ LLM move sequence response format incorrect.")
                    continue
                scene_description = lines2[0]
                move_sequence_line = lines2[1]
                move_sequence = [m.strip() for m in move_sequence_line.split(',') if m.strip()]
                print(f" Scene description: {scene_description}")
                print(f"🔨 Moves: {move_sequence}")
                task = 'bridge' if selected_object_id == 1 else 'tunnel'
            except Exception as e:
                print(f" OpenAI API Error (move sequence): {e}")
                continue

            # --- Process LLM moves ---
            
            obs, info, llm_movement_log = process_llm_moves(
                move_sequence=move_sequence, 
                task=task, 
                env=env, 
                agent=agent, 
                obs=final_obs, 
                info=info, 
                show_images=show_construction_images
            )
            print(f" LLM move sequence processed. Log: {llm_movement_log}")

            # Optionally, visualize or further analyze here

            # --- Plot final state after building bridge/tunnel ---
            print(f"\n FINAL STATE - After building {task}:")
            obs_uint8 = obs.astype(np.uint8)
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(obs_uint8)
            plt.title(f"Final Visual Image (After {task})", fontsize=14)
            plt.axis('off')

            # Extract FOV for plotting
            final_semantic_map = info.get('semantic', np.zeros((64, 64), dtype=np.int32))
            final_player_pos = info.get('player_pos', np.array([32, 32]))
            final_fov_semantic, final_player_fov_pos, _ = extract_player_fov_semantic(
                final_semantic_map, final_player_pos, 0, view_rows=36, view_cols=36, item_rows=8)

            plt.subplot(1, 2, 2)
            plt.imshow(final_fov_semantic, cmap=crafter_colormap, vmin=0, vmax=19)
            plt.title("Final FOV Semantic Map", fontsize=14)
            plt.colorbar(label='Material/Object ID', shrink=0.8)
            plt.plot(final_player_fov_pos[1], final_player_fov_pos[0], 'r*', markersize=20,
                     markeredgecolor='white', markeredgewidth=2, label='Player')
 
            plt.tight_layout()
            plt.show()

            # continue  # Allow periodic LLM calls

        if term or trunc:
            print(f" Environment terminated at step {step}")
            break

    # Only return the last state if needed, no more LLM/semantic/visualization code here
    if 'final_semantic_pov' in locals() and 'center_pos' in locals():
        return final_semantic_pov, center_pos
    elif 'fov_semantic' in locals() and 'player_fov_pos' in locals():
        return fov_semantic, player_fov_pos
    else:
        return None, None
