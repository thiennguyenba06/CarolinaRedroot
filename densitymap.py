import georef2
import numpy as np
import pyexiv2
import os
from shapely.geometry import Polygon, box, Point
import csv

# setting up paths
ORIGIN_PATH = "DJI_202508081433_021_PineIslandbog5H3m5x3photo/DJI_20250808143604_0001_D_Waypoint1.JPG"
IMG_DIR = "DJI_202508081433_021_PineIslandbog5H3m5x3photo"
LABEL_DIR = "output"

# ORIGIN_PATH = "DJI_202507011146_136_PineIslandbog9H3m3x3photo/DJI_20250701114908_0001_D_Waypoint1.JPG"
# IMG_DIR = "DJI_202507011146_136_PineIslandbog9H3m3x3photo"
# LABEL_DIR = "output"
# setting up constants
SIDE_LENGTH_METERS = 1 # grid square side length in meters
yaw = np.radians(float(pyexiv2.Image(ORIGIN_PATH).read_xmp()['Xmp.drone-dji.FlightYawDegree']))

# populate map
all_detections_coor = []


label_list = [] # list of label file paths
img_list = [] # list of image file paths

img_list = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")])[1:]
label_list = sorted([f for f in os.listdir(LABEL_DIR) if f.lower().endswith(".txt")])

print("______________________")
print("Image list: ")
print(img_list)
print("\n")
print("Label list: ")
print(label_list)
print("______________________")

# mapping detections to relative coordinate with drone's first image as basis
# y direction is drone's forward direction
# x direction is always orthogonal to y direction to the right
detections = {} # image_id -> list of (x,y) detections in relative coordinate system
img_fname_map = {} # image_id -> image file name


for img, label in zip(img_list, label_list):
    img_path = os.path.join(IMG_DIR, img)
    img_id = int(img.split("Waypoint")[1].split(".")[0])
    img_fname_map[img_id] = img
    label_path = os.path.join(LABEL_DIR, label)
    mapped_list = georef2.georef(ORIGIN_PATH, img_path, label_path)
    all_detections_coor.extend(mapped_list)
    detections[img_id] = mapped_list
all_detections_coor = np.array(all_detections_coor)




# Write to text file
with open("all_detections.txt", "w") as f:
    for x, y in all_detections_coor:
        f.write(f"{x:.3f} {y:.3f}\n")
# finished mapping detections to relative coordinate with drone's first image as basis


# Image corners
corners_dict = {}  # image_id -> list of corners in (x,y) relative to origin image -- 2d numpy array
image_bounds = {}
for image_f in img_list:
    image_path = os.path.join(IMG_DIR, image_f)
    img_id = int(image_f.split("Waypoint")[1].split(".")[0])
    corners = georef2.get_image_corners(ORIGIN_PATH, image_path)
    corners_dict[img_id] = corners
    # print(corners)
    image_bounds[img_id] = Polygon(corners)

    
# create grids
x_min, x_max = np.min(all_detections_coor[:,0]), np.max(all_detections_coor[:,0])
y_min, y_max = np.min(all_detections_coor[:,1]), np.max(all_detections_coor[:,1])


# number of cells in x and y direction
num_x_cells = int(np.ceil((x_max - x_min) / SIDE_LENGTH_METERS))
print(f"Number of cells in x direction: {num_x_cells}")
num_y_cells = int(np.ceil((y_max - y_min) / SIDE_LENGTH_METERS))
print(f"Number of cells in y direction: {num_y_cells}")


# create array of grid
density_grid = np.zeros((num_y_cells, num_x_cells), dtype=int) # a matrix of dimension num_y_cells x num_x_cells initialized to 0 
point_cell_map = {} # (x,y) point -> image to avoid doulbe counting

# create grid lines
y_lines = np.linspace(start=y_max, stop=y_min, num=num_y_cells+1)

x_lines = np.linspace(start=x_min, stop=x_max, num=num_x_cells+1)


print("Populating density grid...")
# helper functions
def cell_center(x_idx, y_idx):
    x = (x_lines[x_idx] + x_lines[x_idx + 1]) / 2
    y = (y_lines[y_idx] + y_lines[y_idx + 1]) / 2
    return x, y

def read_gps(image_path):
    with pyexiv2.Image(image_path) as img:
        meta = img.read_xmp()
        lat = float(meta['Xmp.drone-dji.GpsLatitude'])
        lon = float(meta['Xmp.drone-dji.GpsLongitude'])
    return lat, lon

def meters_to_gps(lat_origin, lon_origin, dx, dy, yaw_angle):
    """
    Approximates new GPS coordinates given an origin and metric offsets (dx, dy).
    Uses a flat-earth approximation suitable for small drone survey areas.
    """
    R_EARTH = 6378137.0  # Earth radius in meters
    
    dx_rotated = dx * np.cos(-yaw_angle) - dy * np.sin(-yaw_angle)
    dy_rotated = dx * np.sin(-yaw_angle) + dy * np.cos(-yaw_angle)
    # Calculate change in latitude
    d_lat = (dy_rotated / R_EARTH) * (180 / np.pi)
    new_lat = lat_origin + d_lat
    
    # Calculate change in longitude (adjusted for latitude)
    d_lon = (dx_rotated  / (R_EARTH * np.cos(np.radians(lat_origin)))) * (180 / np.pi)
    new_lon = lon_origin + d_lon
    
    return (new_lat, new_lon)

# origin GPS coor
lat0, lon0 = read_gps(ORIGIN_PATH)


gps_map = {}



for y_idx in range(num_y_cells):
    for x_idx in range(num_x_cells):
        up = y_lines[y_idx+1]
        down = y_lines[y_idx]
        left = x_lines[x_idx]
        right = x_lines[x_idx+1]
        cell_bounds = box(left, down, right, up)

        most_count = 0
        chosen_image = None
        for img_id, img_bounds in image_bounds.items():
            if img_bounds.intersects(cell_bounds):
                points = detections[img_id]
                count = sum(1 for p in points if cell_bounds.contains(Point(p[0], p[1])))

                if count > most_count:
                    most_count = count
                    chosen_image = img_id
        
        normalized_val = most_count / SIDE_LENGTH_METERS**2  # density per square meter
        density_grid[y_idx, x_idx] = normalized_val
        if chosen_image is not None:
            point_cell_map[(x_idx, y_idx)] = chosen_image
            # map cell center to GPS
            cell_center_x = (left + right) / 2
            cell_center_y = (down + up) / 2
            gps = meters_to_gps(lat0, lon0, cell_center_x, cell_center_y, yaw)
            gps_map[gps] = (normalized_val, img_fname_map[chosen_image])
        

# output
print("Total images with detections:", len(image_bounds))
print("Total detection files loaded:", len(detections))
print("Nonzero grid cells:", np.count_nonzero(density_grid))
print("Max density in a cell: ", np.max(density_grid))
print(f"Origin GPS: lat {lat0}, lon {lon0}")

with open("gps_density_dict.txt", "w") as f:
    for k, v in gps_map.items():
        f.write(f"{k}: {v}\n" + "\n")
    f.close()



# output file name
csv_output = "gps_density_results.csv"

with open(csv_output, "w", newline='') as f:
    writer = csv.writer(f)

    writer.writerow(["latitude", "longitude", "density", "image_id"])
    
    for (lat, lon), (density, image_fname) in gps_map.items():
        writer.writerow([lat, lon, density, image_fname])

print(f"Data saved for QGIS in {csv_output}")
print("Done.")