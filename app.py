import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches
import numpy as np

def normalize(value, min_val=0, max_val=100):
    """Normalize value to 0–100 scale"""
    if max_val <= min_val:
        return 0  # Avoid division by zero if min and max are the same
    return min(100, max(0, (value - min_val) / (max_val - min_val) * 100))

def calculate_walkability_score(location, radius=1000):
    """
    Calculate walkability score for a location based on:
    - Number of amenities within walking distance
    - Street network density
    - Intersection density
    """
    try:
        # Download street network and amenities
        G = ox.graph_from_point(location, dist=radius, network_type='walk')
        if G is None or len(G) == 0:
            print("Error: Could not retrieve street network data. Check location and/or internet connection.")
            return None, None

        # Get amenities from OSM
        amenities = ox.features_from_point(
            location,
            tags={'amenity': True},
            dist=radius
        )

        # Calculate metrics
        area = radius * radius * 3.14159 / 1000000  # km²

        # Street density (km/km²)
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        street_density = edges['length'].sum() / 1000 / area

        # Intersection density
        nodes, edges = ox.graph_to_gdfs(G)
        intersection_density = len(nodes[nodes['street_count'] > 1]) / area

        # Amenity score
        amenity_score = len(amenities) / area

        # Calculate final score (weighted average)
        walkability_score = (
            0.4 * normalize(street_density) +
            0.4 * normalize(intersection_density) +
            0.2 * normalize(amenity_score)
        )

        return walkability_score, {
            'street_density': street_density,
            'intersection_density': intersection_density,
            'amenities_per_sqkm': amenity_score
        }

    except Exception as e:
        print(f"An error occurred during walkability calculation: {e}")
        return None, None


def plot_walkability(location, radius=1000):
    """Plots walkability with enhanced aesthetics and information."""
    result, metrics = calculate_walkability_score(location, radius)
    if result is None:
        print("Skipping plot due to errors in score calculation.")
        return

    walkability_score = result

    # Download street network for plotting
    G = ox.graph_from_point(location, dist=radius, network_type='walk')
    if G is None or len(G) == 0:
        print("Error: Could not retrieve street network for plotting.")
        return

    # Get amenities for plotting
    amenities = ox.features_from_point(location, tags={'amenity': True}, dist=radius)

    # --- Create a Figure and Axes Grid ---
    fig = plt.figure(figsize=(18, 10))  # Adjust overall figure size
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1,1])  # 2 rows, 3 cols

    # --- MAP Plotting ---
    ax_map = fig.add_subplot(gs[:, 0])  # Span all rows, first column
    ax_map.set_facecolor('#e8f0f7')  # Soft background color

    # Plot street network
    ec = ox.plot.get_edge_colors_by_attr(G, attr='length', cmap='Blues', num_bins=10)
    lc = '#333333'  # Darker color for the location marker border
    ox.plot_graph(G, ax=ax_map, node_size=0, edge_color=ec, edge_linewidth=0.8,
                 bgcolor='#e8f0f7', show=False, close=False, edge_alpha=0.7)

    # Plot amenities
    if not amenities.empty:
        amenities.plot(ax=ax_map, color='orangered', markersize=30, alpha=0.8,
                       label='Amenities', edgecolor='white', linewidth=0.5)

    # Plot the location marker
    ax_map.scatter(location[1], location[0], color='dodgerblue', s=150, marker='o',
               label='Location', edgecolor=lc, linewidth=.5, zorder=5)

    ax_map.set_title(f"Walkability Analysis\nWalkability Score: {walkability_score:.2f}",
                 fontsize=16, fontweight='bold', color='#333333')
    ax_map.set_xlabel(f"Analysis within {radius} meters radius", fontsize=10, color='#555555')
    legend = ax_map.legend(fontsize=11, loc='upper right', frameon=True, facecolor='white', edgecolor='lightgray')

    # Add scale bar and north arrow
    scalebar = ScaleBar(1, 'm', location='lower left', length_fraction=0.25,
                        frameon=True, color='#555555', box_color='lightgray', box_alpha=0.5)
    ax_map.add_artist(scalebar)
    x, y, arrow_length = 0.95, 0.95, 0.05
    ax_map.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=1, headwidth=5),
                ha='center', va='center', fontsize=12, xycoords='axes fraction')

    # Remove axis ticks and labels
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)


    # --- BAR CHART Plotting ---
    ax_bar = fig.add_subplot(gs[0, 1])  # First row, second column

    metrics_labels = ["Street Density", "Intersection Density", "Amenities per Sqkm"]
    metric_values = [metrics['street_density'], metrics['intersection_density'], metrics['amenities_per_sqkm']]

    ax_bar.bar(metrics_labels, [normalize(value) for value in metric_values], color=['#66b3ff', '#99ff99', '#ffcc99'], alpha=0.7)
    ax_bar.set_title("Metric Scores (Normalized)", fontsize=12, color='#333333')
    ax_bar.set_ylabel("Score (0-100)", fontsize=10, color='#555555')
    ax_bar.tick_params(axis='x', labelrotation=45, labelsize=9)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.6)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_facecolor('#f0f0f0')

    # --- Gauge Plotting ---
    ax_gauge = fig.add_subplot(gs[0,2]) # First row, third column

    # Gauge parameters
    gauge_range = 100  # Max score
    num_segments = 10  # Number of segments in the gauge
    segment_values = np.linspace(0, gauge_range, num_segments + 1)
    colors = ['#f73e4b', '#f75e3e', '#f7803e', '#f7a13e', '#f7c13e', '#f7dd3e', '#f7f13e', '#bdf73e', '#71f73e', '#4ef744']

    # Create the gauge segments
    wedge_width = 0.2 # Width of gauge segments
    theta_start = 180
    theta_end = 0


    center_circle = plt.Circle((0,0),0.6,fc='#f0f0f0')
    ax_gauge.add_artist(center_circle)
    for i in range(len(segment_values)-1):
        # Calculate start and end angles for the segment
        start_angle = theta_start - (theta_start-theta_end)*(segment_values[i]/gauge_range)
        end_angle = theta_start - (theta_start-theta_end)*(segment_values[i+1]/gauge_range)

        # Create the wedge patch
        wedge = patches.Wedge((0,0), 1, start_angle, end_angle, width=wedge_width, facecolor=colors[i], edgecolor='white')
        ax_gauge.add_patch(wedge)


    # Add the arrow
    arrow_angle =  theta_start - (theta_start-theta_end)*(walkability_score/gauge_range)
    x = 1.02 * np.cos(np.deg2rad(arrow_angle))
    y = 1.02 * np.sin(np.deg2rad(arrow_angle))
    ax_gauge.arrow(0, 0, x, y, width=0.02, head_width=0.05, head_length=0.1, length_includes_head=True, fc='black', ec='black')

    # Set the center text with score
    ax_gauge.text(0, 0.03, f"{walkability_score:.0f}", fontsize=20, ha='center', va='center', fontweight='bold', color='#333333')

    ax_gauge.set_title("Overall Walkability Score", fontsize=12, color='#333333', pad=15) # Adjusted title
    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(-1.2, 1.2)
    ax_gauge.set_aspect('equal')
    ax_gauge.axis('off')
    ax_gauge.set_facecolor('#f0f0f0')


    plt.tight_layout()
    plt.show()

# Example Usage
location = (24.703976, 46.688154)
plot_walkability(location)
