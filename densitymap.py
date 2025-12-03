import georef2
import numpy as np
import pyexiv2
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, Point

# with open("./all_detections.txt", "w") as f:
#     for coor in all_detections_coor:
#         to_write = str(coor[0]) + ", " + str(coor[1]) + "\n"
#         f.write(to_write)
#     f.close()


# setting up paths
ORIGIN_PATH = "DJI_202508081433_021_PineIslandbog5H3m5x3photo/DJI_20250808143604_0001_D_Waypoint1.JPG"
IMG_DIR = "DJI_202508081433_021_PineIslandbog5H3m5x3photo"
LABEL_DIR = "output"

# setting up constants
SIDE_LENGTH_METERS = 3  # grid square side length in meters

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
detections = {} # image_id -> list of (x,y) detections in relative coordinate system
for img, label in zip(img_list, label_list):
    img_path = os.path.join(IMG_DIR, img)
    img_id = int(img.split("Waypoint")[1].split(".")[0])
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
y_lines = np.linspace(start=y_min, stop=y_max, num=num_y_cells+1)
if abs(x_min) > abs(x_max):
    x_lines = np.linspace(start=x_max, stop=x_min, num=num_x_cells+1)
else:
    x_lines = np.linspace(start=x_min, stop=x_max, num=num_x_cells+1)
all_detections_coor_copy = all_detections_coor.copy()
# populate density grid
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
        
        density_grid[y_idx, x_idx] = most_count
        if chosen_image is not None:
            point_cell_map[(x_idx, y_idx)] = chosen_image

# output
print("Total images with detections:", len(image_bounds))
print("Total detection files loaded:", len(detections))
print("Nonzero grid cells:", np.count_nonzero(density_grid))
print("Max density in a cell: ", np.max(density_grid))