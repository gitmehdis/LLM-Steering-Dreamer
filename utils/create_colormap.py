def create_colormap():

    """Create an intuitive colormap for Crafter semantic maps"""
    import matplotlib.colors as mcolors
    
    # Define intuitive colors for each material/object ID
    # Using RGB values between 0-1
    color_mapping = {
        0: [0.0, 0.0, 0.0],      # None/empty -> Black
        1: [0.0, 0.4, 0.8],      # Water -> Blue
        2: [0.2, 0.8, 0.2],      # Grass -> Green  
        3: [0.5, 0.5, 0.5],      # Stone -> Gray
        4: [0.8, 0.7, 0.4],      # Path -> Light brown/tan
        5: [0.9, 0.8, 0.6],      # Sand -> Beige/sandy
        6: [0.0, 0.6, 0.0],      # Tree -> Dark green
        7: [1.0, 0.2, 0.0],      # Lava -> Red/orange
        8: [0.2, 0.2, 0.2],      # Coal -> Dark gray/black
        9: [0.7, 0.7, 0.8],      # Iron -> Light metallic gray
        10: [0.4, 0.8, 1.0],     # Diamond -> Light blue/cyan
        11: [0.6, 0.4, 0.2],     # Table -> Brown
        12: [0.8, 0.3, 0.1],     # Furnace -> Dark orange/brown
        13: [1.0, 1.0, 0.0],     # Player -> Bright yellow
        14: [1.0, 0.8, 0.6],     # Cow -> Light brown/tan
        15: [0.3, 0.7, 0.3],     # Zombie -> Sickly green
        16: [0.9, 0.9, 0.9],     # Skeleton -> White/bone
        17: [0.8, 0.6, 0.4],     # Arrow -> Brown
        18: [0.4, 0.9, 0.4],     # Plant -> Light green
    }
    
    # Create a colormap with 20 colors (covering all possible IDs)
    colors = []
    for i in range(20):
        if i in color_mapping:
            colors.append(color_mapping[i])
        else:
            colors.append([0.8, 0.8, 0.8])  # Default light gray for unknown IDs
    
    # Create custom colormap
    crafter_cmap = mcolors.ListedColormap(colors, name='crafter')
    return crafter_cmap

    # Create the custom colormap
    crafter_colormap = create_intuitive_colormap()

    print("✅ Created intuitive colormap for Crafter materials and objects!")
    return crafter_colormap