import georef2
import numpy as np
import pyexiv2
import os
from shapely import Polygon, box
from shapely import STRtree
import csv
from concurrent.futures import ProcessPoolExecutor

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
origin_gps = (float(pyexiv2.Image(ORIGIN_PATH).read_xmp()['Xmp.drone-dji.GpsLatitude']), float(pyexiv2.Image(ORIGIN_PATH).read_xmp()['Xmp.drone-dji.GpsLongitude']))


img_list = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")])[1:]
label_list = sorted([f for f in os.listdir(LABEL_DIR) if f.lower().endswith(".txt")])


# mapping detections to relative coordinate with drone's first image as basis
# y direction is drone's forward direction
# x direction is always orthogonal to y direction to the right
detections = {} # image_id -> list of (x,y) detections in relative coordinate system
img_fname_map = {} # image_id -> image file name
image_bounds = {}
all_detections_coor = [] # list of all (x,y) detections in relative coordinate system for all images, used to determine grid size and bounds

gps_map = {} # (lat, lon) -> (density, image_fname) mapping for each grid cell center


def process_img(img, label):
    img_path = os.path.join(IMG_DIR, img)
    img_id = int(img.split("Waypoint")[1].split(".")[0])
    label_path = os.path.join(LABEL_DIR, label)
    img_id = int(img.split("Waypoint")[1].split(".")[0])

    mapped_list = georef2.georef(ORIGIN_PATH, img_path, label_path)
    polygon = Polygon(georef2.get_image_corners(ORIGIN_PATH, img_path))

    return {
        "img": img,
        "img_id": img_id,
        "img_path": img_path,
        "mapped_list": mapped_list,
        "polygon": polygon
    }

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



if __name__ == "__main__":
    # multithreading for image processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_img, img_list, label_list))

    for result in results:
        all_detections_coor.extend(result["mapped_list"])
        detections[result["img_id"]] = result["mapped_list"]
        image_bounds[result["img_id"]] = result["polygon"]
        img_fname_map[result["img_id"]] = result["img"]
        
    all_detections_coor = np.array(all_detections_coor) # convert list to numpy array for easier processing later
    detections = {img_id: np.array(weed) for img_id, weed in detections.items()} # convert lists to numpy arrays for easier processing later

    id_list = np.sort(np.array(list(image_bounds.keys())))
    img_bounds_ordered = np.array([image_bounds[img_id] for img_id in id_list])
    tree = STRtree(img_bounds_ordered)
    if tree is not None:
        print("Successfully built spatial index for image bounds.")
    else: 
        print("Failed to build spatial index for image bounds.")
        raise Exception("Spatial index construction failed.")
    

    print("Finished processing images and mapping detections to relative coordinates with origin of drone's first image. \n")

    # create grids
    x_min, x_max = np.min(all_detections_coor[:,0]), np.max(all_detections_coor[:,0])
    y_min, y_max = np.min(all_detections_coor[:,1]), np.max(all_detections_coor[:,1])

    # number of cells in x and y direction
    # y is the drone's forward direction, x is the right direction orthogonal to y
    num_x_cells = int(np.ceil((x_max - x_min) / SIDE_LENGTH_METERS))
    print(f"Number of cells in x direction: {num_x_cells}")
    num_y_cells = int(np.ceil((y_max - y_min) / SIDE_LENGTH_METERS))
    print(f"Number of cells in y direction: {num_y_cells} \n")

    # create cells
    density_grid = np.zeros((num_y_cells, num_x_cells), dtype=int) # a matrix of dimension num_y_cells x num_x_cells initialized to 0 

    # create grid lines
    y_lines = np.linspace(start=y_min, stop=y_max, num=num_y_cells+1)
    x_lines = np.linspace(start=x_min, stop=x_max, num=num_x_cells+1)


    print("Density calculation started...")
    # density calculation
    for y_idx in range(num_y_cells):
        for x_idx in range(num_x_cells):
            up, down = y_lines[y_idx+1], y_lines[y_idx]
            left, right = x_lines[x_idx], x_lines[x_idx+1]
            cell_bounds = box(left, down, right, up)

            possible_bounds = tree.query(cell_bounds) # 1d array of indices of possible intersecting polygons

            # debug purpose
            # if (len(possible_bounds) != 0):
            #     print(possible_bounds)

            most_count = 0
            chosen_img = None
            for idx in possible_bounds:
                img_id = id_list[idx]
                points = detections[img_id]

                is_in_x = (points[:,0] >= left) & (points[:,0] <= right)
                is_in_y = (points[:,1] >= down) & (points[:,1] <= up)

                count = np.count_nonzero(is_in_x & is_in_y)

            
                if count > most_count:
                    most_count = count
                    chosen_img = img_id
            
            if chosen_img is not None:
                density = most_count / SIDE_LENGTH_METERS**2  # density per square meter
                density_grid[y_idx, x_idx] = density

                # map cell center to GPS
                cell_center_x = (left + right) / 2
                cell_center_y = (down + up) / 2
                gps = meters_to_gps(origin_gps[0], origin_gps[1], cell_center_x, cell_center_y, yaw)
                gps_map[gps] = (density, img_fname_map[chosen_img])

    # output
    print("Total images with detections:", len(image_bounds))
    print("Total detection files loaded:", len(detections))
    print("Nonzero grid cells:", np.count_nonzero(density_grid))
    print("Max density in a cell: ", np.max(density_grid))
    print(f"Origin GPS: lat {origin_gps[0]}, lon {origin_gps[1]}\n")

    # output file name
    csv_output = "gps_density_results.csv"

    with open(csv_output, "w", newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["latitude", "longitude", "density", "image_id"])

        for (lat, lon), (density, image_fname) in gps_map.items():
            writer.writerow([lat, lon, density, image_fname])

    print(f"Data saved for QGIS in {csv_output}")
    print("Done.")