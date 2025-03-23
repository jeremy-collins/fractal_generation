#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
import numpy as np
import time
import torch
import math

# Configuration dictionary with all settings
config = {
    # Visualization settings
    'resolution': 1000,           # Grid resolution (higher = more detailed but slower)
    'figsize': (24, 24),         # Size of the figure in inches
    
    # Mandelbrot iteration settings
    'base_min_depth': 3,         # Minimum iteration depth for initial view
    'base_max_depth': 100,       # Maximum iteration depth for initial view
    'depth_step': 10,            # Step size between depths
    'adaptive_depth': True,      # Whether to adjust iteration depth based on zoom level
    'depth_scale_factor': 10,    # How quickly to increase depth with zoom (higher = more depth increase)
    
    # Initial view coordinates
    'initial_view': {
        'min_x': -2.0,
        'max_x': 1.0,
        'min_y': -1.25,
        'max_y': 1.25
    },
    
    # # Alternative zoomed view (uncomment to use)
    # 'initial_view': {
    #     'min_x': -0.7585,
    #     'max_x': -0.7550,
    #     'min_y': 0.0610, 
    #     'max_y': 0.0645
    # },
    
    # GPU settings
    'batch_size': 1000000,       # Number of points to process at once (adjust based on GPU memory)
    
    # Visual settings
    'background_color': 'xkcd:blue',
    'selection_color': 'r',      # Color of selection rectangle
    'selection_linewidth': 2,    # Width of selection rectangle border
    
    # Colormap settings
    'colormap': {
        'type': 'matplotlib',    # 'hsv', 'matplotlib', or 'custom'
        'matplotlib_name': 'twilight_shifted',  # Any standard matplotlib colormap name
                                  # Try: viridis, plasma, inferno, magma, cividis, 
                                  # coolwarm, jet, rainbow, ocean, terrain, etc.
                                  # Add '_r' to the name for reversed version (e.g., 'viridis_r')
        'reverse': False,        # Another way to reverse the colormap direction
        
        # HSV colormap settings (used when type='hsv')
        'hsv_hue_start': 0.7,    # Starting hue (0-1) for HSV colormap
        'hsv_saturation': 1.0,   # Saturation (0-1) for HSV colormap
        
        # Custom colormap as list of (depth_fraction, color) tuples
        # depth_fraction ranges from 0 (min_depth) to 1 (max_depth)
        'custom_colors': [
            (0.0, 'darkblue'),   # For min_depth
            (0.3, 'blue'),
            (0.5, 'purple'),
            (0.7, 'crimson'),
            (0.9, 'orange'),
            (1.0, 'gold')        # For max_depth
        ]
    },
    
    # Output settings
    'save_final_image': True,
    'output_filename': 'mandelbrot_final.png',
    
    # Utility settings
    'list_colormaps': False,     # Set to True to print all available colormaps at startup
}

# Initialize figure and tracking variables
fig = plt.figure(figsize=config['figsize'])
zoom_coords = []
current_rect = None  # Store the current rectangle being drawn

def get_color_for_depth(depth, min_depth, max_depth):
    """
    Get color for a specific iteration depth based on the configuration.
    """
    # Calculate depth as fraction of max_depth for color mapping
    depth_fraction = (depth - min_depth) / (max_depth - min_depth)
    
    # Adjust direction if colormap is reversed
    if config['colormap'].get('reverse', False):
        depth_fraction = 1.0 - depth_fraction
    
    if config['colormap']['type'] == 'hsv':
        # Use HSV colormap
        hue = config['colormap']['hsv_hue_start'] * (1 - depth_fraction)
        saturation = config['colormap']['hsv_saturation']
        value = 1 - depth_fraction
        return hsv_to_rgb([hue, saturation, value])
    
    elif config['colormap']['type'] == 'matplotlib':
        # Use standard matplotlib colormap
        cmap_name = config['colormap']['matplotlib_name']
        cmap = plt.cm.get_cmap(cmap_name)
        return cmap(depth_fraction)
    
    elif config['colormap']['type'] == 'custom':
        # Use custom colormap with interpolation
        custom_colors = config['colormap']['custom_colors']
        
        # Find the color stops that bound our depth_fraction
        for i in range(len(custom_colors) - 1):
            start_frac, start_color = custom_colors[i]
            end_frac, end_color = custom_colors[i + 1]
            
            if start_frac <= depth_fraction <= end_frac:
                # Convert color names to RGB if needed
                if isinstance(start_color, str):
                    start_color = np.array(plt.cm.colors.to_rgb(start_color))
                if isinstance(end_color, str):
                    end_color = np.array(plt.cm.colors.to_rgb(end_color))
                
                # Interpolate between the two colors
                interp_factor = (depth_fraction - start_frac) / (end_frac - start_frac)
                return start_color + interp_factor * (end_color - start_color)
        
        # Fallback to the last color if something went wrong
        last_color = custom_colors[-1][1]
        if isinstance(last_color, str):
            return plt.cm.colors.to_rgb(last_color)
        return last_color

def calculate_depth_range(zoom_factor):
    """
    Calculate appropriate iteration depth range based on zoom level.
    Higher zoom levels need more iterations to see detail.
    """
    if not config['adaptive_depth'] or zoom_factor <= 1:
        return config['base_min_depth'], config['base_max_depth']
    
    # Calculate logarithmic scaling for depth
    log_zoom = math.log10(max(1, zoom_factor))
    depth_increase = int(log_zoom * config['depth_scale_factor'])
    
    min_depth = config['base_min_depth']
    max_depth = config['base_max_depth'] + depth_increase
    
    print(f"Zoom factor {zoom_factor:.1f}x → Depth range: {min_depth}-{max_depth}")
    
    return min_depth, max_depth

def check_mandelbrot_gpu(real_batch, imag_batch, max_iter, device, dtype=torch.float64):
    """
    Check if points are in the Mandelbrot set using PyTorch on GPU with arbitrary precision.
    Implements both the iteration test and the cardioid/bulb test.
    """
    batch_size = real_batch.size(0)
    
    # Initialize result
    in_set = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Test for cardioid and period-2 bulb
    # Calculate magnitude and phase for each point
    c_mag = torch.sqrt(real_batch**2 + imag_batch**2)
    c_phase = torch.atan2(imag_batch, real_batch)
    
    # Cardioid test
    cos_phase_half = torch.cos(c_phase/2)
    sin_phase_half = torch.sin(c_phase/2)
    cos_phase_quarter = torch.cos(c_phase/4)
    sin_phase_quarter = torch.sin(c_phase/4)
    
    exp_half_real = cos_phase_half
    exp_half_imag = sin_phase_half
    exp_quarter_real = cos_phase_quarter
    exp_quarter_imag = sin_phase_quarter
    
    diff_real = exp_half_real - exp_quarter_real
    diff_imag = exp_half_imag - exp_quarter_imag
    cardioid_bound = torch.sqrt(diff_real**2 + diff_imag**2)
    
    in_cardioid = c_mag < cardioid_bound
    
    # Circle test
    circle_center_real = -1.0
    circle_center_imag = 0.0
    
    circle_radius = 0.25 * torch.sqrt(cos_phase_half**2 + sin_phase_half**2)
    
    dist_from_center = torch.sqrt(
        (real_batch - circle_center_real)**2 + 
        (imag_batch - circle_center_imag)**2
    )
    
    in_circle = dist_from_center < circle_radius
    
    # Mark points in cardioid or circle as in the set
    in_set = in_set | in_cardioid | in_circle
    
    # For points not in known regions, do the iteration test
    not_in_known_regions = ~(in_cardioid | in_circle)
    
    # Skip iteration if all points are already determined
    if torch.any(not_in_known_regions):
        # Get points that need iteration
        test_indices = torch.nonzero(not_in_known_regions, as_tuple=False).view(-1)
        test_real = real_batch[test_indices]
        test_imag = imag_batch[test_indices]
        
        # Initialize z
        z_real = torch.zeros_like(test_real)
        z_imag = torch.zeros_like(test_imag)
        
        # Track which points have escaped
        escaped = torch.zeros(len(test_real), dtype=torch.bool, device=device)
        
        # Iterate
        for i in range(max_iter):
            # Only compute for non-escaped points
            non_escaped_mask = ~escaped
            
            # Check if all points have escaped
            if not torch.any(non_escaped_mask):
                break
                
            # Get indices of non-escaped points
            compute_indices = torch.nonzero(non_escaped_mask, as_tuple=False).view(-1)
                
            # z = z^2 + c for non-escaped points
            z_real_masked = z_real[compute_indices]
            z_imag_masked = z_imag[compute_indices]
            
            # z^2
            z_real_sq = z_real_masked**2 - z_imag_masked**2
            z_imag_sq = 2 * z_real_masked * z_imag_masked
            
            # z^2 + c
            z_real[compute_indices] = z_real_sq + test_real[compute_indices]
            z_imag[compute_indices] = z_imag_sq + test_imag[compute_indices]
            
            # Check for escape
            magnitude = z_real[compute_indices]**2 + z_imag[compute_indices]**2
            new_escaped = magnitude > 4.0  # |z|^2 > 4 is equivalent to |z| > 2
            
            # Mark newly escaped points
            if torch.any(new_escaped):
                newly_escaped_indices = compute_indices[new_escaped]
                escaped[newly_escaped_indices] = True
        
        # Points that never escaped are in the set
        test_results = ~escaped
        
        # Update the original in_set tensor for these points
        in_set_update_indices = test_indices[test_results]
        if len(in_set_update_indices) > 0:
            in_set[in_set_update_indices] = True
    
    # Extract points that are in the set
    in_set_indices = torch.nonzero(in_set, as_tuple=False).view(-1)
    
    if len(in_set_indices) > 0:
        # Get coordinates of points in the set
        points_in_set = torch.stack([real_batch[in_set_indices], imag_batch[in_set_indices]], dim=1)
        return points_in_set
    else:
        return torch.empty((0, 2), device=device)

def mandelbrot_vectorized(h, w, xmin, xmax, ymin, ymax, max_iter):
    """
    GPU-accelerated implementation of the Mandelbrot set calculation using PyTorch.
    Uses scaled coordinates for arbitrary precision.
    
    Parameters:
    h, w -- Height and width of the grid
    xmin, xmax, ymin, ymax -- Coordinate boundaries
    max_iter -- Maximum iteration depth
    """
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Always use double precision for more accurate results
    torch_dtype = torch.float64
    print(f"Using double precision with arbitrary scaling")
    
    start_time = time.time()
    
    # Calculate the center of the view for reference point
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # Calculate the width and height of the view
    width = xmax - xmin
    height = ymax - ymin
    
    # Create normalized coordinate grid from -1 to 1
    norm_x = torch.linspace(-1, 1, w, dtype=torch_dtype, device=device)
    norm_y = torch.linspace(-1, 1, h, dtype=torch_dtype, device=device)
    
    # Create meshgrid for the normalized coordinates
    norm_X, norm_Y = torch.meshgrid(norm_x, norm_y, indexing='xy')
    
    # Scale the normalized coordinates to the actual width and height
    # These are offsets from the center point
    scaled_X = center_x + norm_X * (width / 2)
    scaled_Y = center_y + norm_Y * (height / 2)
    
    # Flatten the coordinates
    real_part = scaled_X.reshape(-1)
    imag_part = scaled_Y.reshape(-1)
    
    # Process all points in batches
    batch_size = min(config['batch_size'], h * w)  # Use a single batch if fits in memory
    num_points = h * w
    num_batches = (num_points + batch_size - 1) // batch_size
    
    print(f"Reference point (center): ({center_x:.15e}, {center_y:.15e})")
    print(f"View dimensions: width={width:.15e}, height={height:.15e}")
    print(f"Processing {num_points} points in {num_batches} batches...")
    
    all_points_in_set = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_points)
        
        # Get batch
        batch_real = real_part[start_idx:end_idx]
        batch_imag = imag_part[start_idx:end_idx]
        
        # Check for the main cardioid and period-2 bulb in batch
        batch_points = check_mandelbrot_gpu(batch_real, batch_imag, max_iter, device, torch_dtype)
        
        if len(batch_points) > 0:
            # Get points back to CPU right away to free GPU memory
            all_points_in_set.append(batch_points.cpu())
    
    # Combine results - do this on CPU to avoid GPU memory issues
    if all_points_in_set:
        points_in_set = torch.cat(all_points_in_set, dim=0).numpy()
    else:
        points_in_set = np.empty((0, 2))
    
    elapsed = time.time() - start_time
    print(f"GPU calculation completed in {elapsed:.2f} seconds. Found {len(points_in_set)} points in set.")
    
    return points_in_set

def plot(min_x, max_x, min_y, max_y):
    """
    Plot the Mandelbrot set with increasing iteration depths for better detail.
    This optimized version minimizes the bottlenecks between GPU calculations.
    """
    plt.clf()
    plt.ioff()
    
    ax = plt.gca()
    ax.set_facecolor(config['background_color'])
    
    # Calculate zoom level for informational purposes
    initial_view = config['initial_view']
    initial_width = min(initial_view['max_x'] - initial_view['min_x'], 
                       initial_view['max_y'] - initial_view['min_y'])
    current_width = min(max_x - min_x, max_y - min_y)
    zoom_factor = initial_width / current_width if current_width > 0 else 0
    
    # Adjust iteration depth based on zoom level
    min_depth, max_depth = calculate_depth_range(zoom_factor)
    
    print(f"\nRendering coordinates: [{min_x:.8f}, {max_x:.8f}] × [{min_y:.8f}, {max_y:.8f}]")
    print(f"Current zoom factor: {zoom_factor:.2f}x")
    
    # Collect all points from different depths before plotting
    all_depths_points = []
    all_depths_colors = []
    
    # Calculate all depths first without plotting
    print("\nPre-calculating all depths...")
    for depth in range(min_depth, max_depth + 1, config['depth_step']):
        print(f"Calculating at depth {depth}... ({depth}/{max_depth})")
        
        # Calculate points in the set at this depth
        points = mandelbrot_vectorized(
            config['resolution'], 
            config['resolution'], 
            min_x, max_x, min_y, max_y, 
            depth
        )
        
        if len(points) > 0:
            # Get color from color mapping function
            color = get_color_for_depth(depth, min_depth, max_depth)
            
            # Store points and their color for later plotting
            all_depths_points.append(points)
            all_depths_colors.append(color)
        else:
            print(f"  No points in set at depth {depth}")
    
    # Now plot all the collected points at once
    print("\nRendering visualization...")
    for i, (points, color) in enumerate(zip(all_depths_points, all_depths_colors)):
        print(f"Rendering depth level {i+1}/{len(all_depths_points)}...")
        plt.plot(points[:, 0], points[:, 1], markersize=(1300/config['resolution']), 
                 color=color, marker="s", linestyle="None")
    
    
    # Save the final image if configured
    if config['save_final_image']:
        print(f"Saving final image to {config['output_filename']}...")
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        # plt.axis('equal')
        plt.axis('off')
        fig.savefig(config['output_filename'])
    
    # Add coordinates to the title
    plt.title(f"Mandelbrot Set: [{min_x:.8f}, {max_x:.8f}] × [{min_y:.8f}, {max_y:.8f}] - Zoom: {zoom_factor:.1f}x")
    print("Showing interactive plot...")
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.axis('equal')
    plt.show()

def onclick(event):
    global zoom_coords, current_rect
    if event.xdata is not None and event.ydata is not None:
        zoom_coords = [(event.xdata, event.ydata)]
        print(f"Starting selection at {event.xdata:.6f}, {event.ydata:.6f}")
        
        # Create a new rectangle at the click position with zero width/height
        current_rect = Rectangle((event.xdata, event.ydata), 0, 0, 
                               linewidth=config['selection_linewidth'], 
                               edgecolor=config['selection_color'], 
                               facecolor='none', zorder=100)
        plt.gca().add_patch(current_rect)
        plt.draw()

def onmove(event):
    global zoom_coords, current_rect
    if len(zoom_coords) == 1 and event.xdata is not None and event.ydata is not None and current_rect is not None:
        # Calculate rectangle dimensions based on starting point and current mouse position
        start_x, start_y = zoom_coords[0]
        width = event.xdata - start_x
        height = event.ydata - start_y
        
        # Update rectangle position and size
        if width >= 0:
            x = start_x
        else:
            x = event.xdata
            width = abs(width)
            
        if height >= 0:
            y = start_y
        else:
            y = event.ydata
            height = abs(height)
        
        # Update the rectangle
        current_rect.set_xy((x, y))
        current_rect.set_width(width)
        current_rect.set_height(height)
        plt.draw()

def onrelease(event):
    global zoom_coords, current_rect
    if event.xdata is not None and event.ydata is not None:
        zoom_coords.append((event.xdata, event.ydata))
        print(f"Ending selection at {event.xdata:.6f}, {event.ydata:.6f}")
        
        if len(zoom_coords) == 2:
            plt.ion()
            xmin = min(zoom_coords[0][0], zoom_coords[1][0])
            xmax = max(zoom_coords[0][0], zoom_coords[1][0])
            ymin = min(zoom_coords[0][1], zoom_coords[1][1])
            ymax = max(zoom_coords[0][1], zoom_coords[1][1])
            
            # Don't need to add a new rectangle as we already have one from dragging
            plt.draw()
            plt.pause(0.1)
            
            # Don't allow zero-sized selections
            x_range = xmax - xmin
            y_range = ymax - ymin
            min_range = min(x_range, y_range)
            
            if min_range < 1e-307:  # Extremely small selection not allowed
                print("\n⚠️ WARNING: Selected region is too small. Please select a larger area.")
                zoom_coords = []
                current_rect = None
                return
                
            # Plot the zoomed region
            plot(xmin, xmax, ymin, ymax)
            zoom_coords = []
            current_rect = None
        else:
            zoom_coords = []
            current_rect = None

def main():
    global zoom_coords
    
    print("\nGPU-Accelerated Mandelbrot Set Visualizer")
    print("========================================")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU detected: {gpu_name}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("⚠️ No GPU detected. Using CPU instead (much slower).")
    
    # Display current colormap setting
    if config['colormap']['type'] == 'hsv':
        print(f"Using HSV colormap with hue starting at {config['colormap']['hsv_hue_start']}")
    elif config['colormap']['type'] == 'matplotlib':
        cmap_name = config['colormap']['matplotlib_name']
        reverse_str = " (reversed)" if config['colormap'].get('reverse', False) else ""
        print(f"Using Matplotlib colormap: '{cmap_name}'{reverse_str}")
    else:
        print(f"Using custom colormap with {len(config['colormap']['custom_colors'])} color stops")
    
    # List available matplotlib colormaps if requested
    if config.get('list_colormaps', False):
        print("\nAvailable Matplotlib colormaps:")
        cmaps = [m for m in plt.cm.datad if not m.endswith("_r")]
        cmaps.sort()
        for i, cmap_name in enumerate(cmaps):
            if i % 5 == 0 and i > 0:
                print()  # New line every 5 colormaps
            print(f"{cmap_name:12}", end=" ")
        print("\n(Add '_r' to any name for reversed version)")
    
    # Set up event handlers
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('motion_notify_event', onmove)  # Connect motion event for live rectangle
    
    # Get initial view coordinates from config
    view = config['initial_view']
    min_x = view['min_x']
    max_x = view['max_x']
    min_y = view['min_y']
    max_y = view['max_y']
    
    # Optional zoomed view (can be toggled by changing the key in the config)
    # Uncomment the next 5 lines to use the zoomed view instead
    # view = config['zoomed_view']
    # min_x = view['min_x']
    # max_x = view['max_x']
    # min_y = view['min_y']
    # max_y = view['max_y']
    
    print("\nInitial render starting... Click and drag to zoom into regions of interest.")
    print(f"The {config['selection_color']} rectangle will update as you drag to show the selected area.")
    print("Adaptive depth scaling is " + ("enabled" if config['adaptive_depth'] else "disabled"))
    
    plot(min_x, max_x, min_y, max_y)

if __name__ == "__main__":
    main()