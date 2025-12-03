import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from collections import defaultdict
import math

# Inputs
image_corners_file = input("Enter path to image corners file: ").strip()
detection_folder = input("Enter path to detection files folder: ").strip()
side_length_meters = float(input("Enter desired grid square side length (in meters): ").strip())

heatmap_image = "gps_field_density_heatmap.png"
geotiff_output = "gps_density_map.tif"
image_id_tif = "image_id_map.tif"
mapping_output = "grid_image_mapping.txt"
legend_output = "image_id_legend.txt"

# Converting to meters
EARTH_RADIUS = 6378137  # meters
DEG_TO_RAD = math.pi / 180


def meters_to_degrees(meters, latitude_deg):
    lat_deg_per_meter = 1 / 111320
    lon_deg_per_meter = 1 / (40075000 * math.cos(latitude_deg * DEG_TO_RAD) / 360)
    return meters * lat_deg_per_meter, meters * lon_deg_per_meter

# Detections
detections = {}
for fname in os.listdir(detection_folder):
    if fname.endswith(".txt"):
        image_id = os.path.splitext(fname)[0]
        path = os.path.join(detection_folder, fname)
        points = []
        with open(path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 8:
                    lats = parts[::2]
                    lons = parts[1::2]
                    lat = sum(lats) / 4
                    lon = sum(lons) / 4
                    points.append((lat, lon))
        detections[image_id] = points

# Image corners
corner_map = defaultdict(dict)
image_polygons = {}
all_lats, all_lons = [], []

with open(image_corners_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        image_id, label, lat, lon = parts
        try:
            lat = float(lat)
            lon = float(lon)
            corner_map[image_id][label] = (lon, lat)
        except ValueError:
            continue



def sort_corners_clockwise(corner_dict):
    points = list(corner_dict.values())
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

for image_id, corners in corner_map.items():
    if len(corners) >= 4:
        sorted_corners = sort_corners_clockwise(corners)
        sorted_corners.append(sorted_corners[0])
        poly = Polygon(sorted_corners)
        if poly.is_valid and poly.is_simple:
            clean_id = os.path.splitext(image_id)[0]
            if clean_id in detections:
                image_polygons[clean_id] = poly
                all_lats.extend([pt[1] for pt in sorted_corners])
                all_lons.extend([pt[0] for pt in sorted_corners])

if not all_lats or not all_lons:
    raise ValueError("No valid image polygons found with matching detection files.")

# Creating grid
lat_min, lat_max = min(all_lats), max(all_lats)
lon_min, lon_max = min(all_lons), max(all_lons)

center_latitude = (lat_min + lat_max) / 2
lat_step_deg, lon_step_deg = meters_to_degrees(side_length_meters, center_latitude)

num_y_grids = max(1, int(np.ceil((lat_max - lat_min) / lat_step_deg)))
num_x_grids = max(1, int(np.ceil((lon_max - lon_min) / lon_step_deg)))

grid_lat_lines = np.linspace(lat_max, lat_min, num_y_grids + 1)
grid_lon_lines = np.linspace(lon_min, lon_max, num_x_grids + 1)

print(f"Calculated grid: {num_x_grids} columns x {num_y_grids} rows")
print(f"Approx grid size: {side_length_meters} meters square")

density_grid = np.zeros((num_y_grids, num_x_grids), dtype=int)
grid_image_map = {}

# Grid calculations
for y_idx in range(num_y_grids):
    for x_idx in range(num_x_grids):
        north = grid_lat_lines[y_idx]
        south = grid_lat_lines[y_idx + 1]
        west = grid_lon_lines[x_idx]
        east = grid_lon_lines[x_idx + 1]
        cell_poly = box(west, south, east, north)

        best_count = 0
        best_image = None
        for image_id, poly in image_polygons.items():
            if poly.contains(cell_poly):
                points = detections[image_id]
                count = sum(
                    1 for lat, lon in points
                    if south <= lat <= north and west <= lon <= east
                )
                if count > best_count:
                    best_count = count
                    best_image = image_id

        density_grid[y_idx, x_idx] = best_count
        if best_image:
            grid_image_map[(y_idx, x_idx)] = best_image

# Checking output
print(" Total images with detections:", len(image_polygons))
print(" Total detection files loaded:", len(detections))
print(" Nonzero grid cells:", np.count_nonzero(density_grid))
print(" Max density value:", np.max(density_grid))

# Heatmap
max_density = np.max(density_grid) or 1
plt.figure(figsize=(8, 6), dpi=300, frameon=False)
plt.imshow(
    density_grid,
    cmap='hot_r',
    norm=mcolors.Normalize(vmin=0, vmax=max_density),
    extent=[lon_min, lon_max, lat_min, lat_max],
    origin='upper',
    alpha=0.8
)
plt.axis('off')
plt.savefig(heatmap_image, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
plt.close()

# Density map
pixel_size_x = (lon_max - lon_min) / num_x_grids
pixel_size_y = (lat_max - lat_min) / num_y_grids
transform = from_origin(lon_min, lat_max, pixel_size_x, pixel_size_y)

with rasterio.open(
    geotiff_output,
    'w',
    driver='GTiff',
    height=density_grid.shape[0],
    width=density_grid.shape[1],
    count=1,
    dtype='float32',
    crs='EPSG:4326',
    transform=transform,
) as dst:
    dst.write(density_grid.astype(np.float32), 1)

print(f" Density GeoTIFF created: {geotiff_output}")

#Image id grid
image_ids = sorted(set(grid_image_map.values()))
image_to_int = {img: i + 1 for i, img in enumerate(image_ids)}
image_id_grid = np.zeros((num_y_grids, num_x_grids), dtype=np.uint16)

for (y, x), image_id in grid_image_map.items():
    image_id_grid[y, x] = image_to_int[image_id]

#Image Ids
with rasterio.open(
    image_id_tif,
    'w',
    driver='GTiff',
    height=image_id_grid.shape[0],
    width=image_id_grid.shape[1],
    count=1,
    dtype='uint16',
    crs='EPSG:4326',
    transform=transform,
) as dst:
    dst.write(image_id_grid, 1)

print(f" Image ID GeoTIFF created: {image_id_tif}")

#Image legend
with open(legend_output, "w") as f:
    for img, val in image_to_int.items():
        f.write(f"{val}: {img}\n")

#Grid images
with open(mapping_output, "w") as f:
    for (y, x), img in grid_image_map.items():
        f.write(f"{y},{x},{img}\n")

print(f" Image ID legend saved to: {legend_output}")
print(f" Grid-to-image mapping saved to: {mapping_output}")
