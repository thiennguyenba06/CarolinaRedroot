import numpy as np
import os
import pyexiv2

SENSOR_FOV_VERTICAL = np.radians(55.072)
SENSOR_FOV_HORIZONTAL = np.radians(69.72)

def GPS_to_Cartesian(tuple):
    """
    a transformation from GPS coordinates in R^3 -> Cartesian coordinates in R^2
    @param tuple: (lat, lon)
    @return: (x, y)
    """
    R = 6371000  # Radius of the Earth in meters
    lat = tuple[0];
    lon = tuple[1];
    x = R * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = R * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    return (x, y)

def change_of_basis(origin, point, basis):
    """
    perform a change of coordinates from standard basis to a new basis
    """
    return None



def find_center(points, width, height):
    """
    Conpute the center of a detection box given its corner points
    @param points: a tuple of corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    @return: x, y
    """
    x_center = sum(point[0] for point in points) / len(points)
    y_center = sum(point[1] for point in points) / len(points)
    # denormalize from [0,1] to pixel coordinates
    x_center = x_center * width
    y_center = y_center * height
    # change of basis: origin at center of image
    x = x_center - width / 2
    y = -y_center + height / 2
    return x, y

def find_angle_y(y, img_height):
    """
    Compute the angle in radians form by the vector from camera to (0,0) and to (0,y)
    param y: y coordinate in pixels
    param img_width: width of the image in pixels
    return: angle in radians
    """
    SO =  img_height / (2*np.tan(SENSOR_FOV_VERTICAL/2))
    angle_y = np.arctan(y/SO)
    return angle_y

def find_angle_x(x, y, img_width):
    """
    Compute the angle in radians form by the vector from camera to (0, y) and to (x, y) 
    param x: x coordinate in pixels
    param y: y coordinate in pixels
    param img_width: width of the image in pixels
    """
    SO = img_width / (2*np.tan(SENSOR_FOV_HORIZONTAL/2))
    #Oy = np.sqrt(SO**2 + y**2)
    angle_x = np.arctan(x/SO)
    return angle_x

def find_point_projection(point, img_width, img_height, drone_height, pitch):
    """
    Find the projection of a point in image coordinates to relative Cartesian coordinates to the drone 
    Mapping: pixels coordinates (origin at center of image) -> cartesian coordinates in meters (origin at drone position)
    param points: center of a detection box
    param img_width: width of the image in pixels
    return : x_distance, y_distance in meters relative to drone position
    """ 
    angle_y = find_angle_y(point[1], img_height)
    angle_x = find_angle_x(point[0], point[1], img_width)
    y_distance = drone_height * np.tan(np.pi/2 + pitch + angle_y)
    x_distance = y_distance * np.tan(angle_x)
    return x_distance, y_distance


def get_detections_coor(img_path, detections_path):
    """
    Get the relative Cartesian coordinates of all detections in an image
    param img_path: path to the image
    param detections_path: path to the detection txt file
    return: list of (x,y) coordinates in meters relative to drone position
    """
    img = pyexiv2.Image(img_path)
    exif = img.read_exif()
    img_width = float(img.read_exif()['Exif.Photo.PixelXDimension'])
    print("img width: ", img_width)
    img_height = float(img.read_exif()['Exif.Photo.PixelYDimension'])
    print("img height: ", img_height)
    coor_list = []
    with open(detections_path) as file:
        lines = file.read().splitlines()
    for line in lines:
        if (line.strip() != ""):
            boxlist = list(line.split(" "))[1:]
            box = [(float(boxlist[0]), float(boxlist[1])), (float(boxlist[2]), float(boxlist[3])),
                   (float(boxlist[4]), float(boxlist[5])), (float(boxlist[6]), float(boxlist[7]))]
            # print(box[0][0])
            coor = find_center(box, img_width, img_height)
            x, y = find_point_projection(coor, img_width, img_height, 3, np.radians(-60))
            # print(coor)
            coor_list.append([x, y])
    return np.array(coor_list)

            
def map_to_origin(new_origin, old_origin, detections_txt):
    """
    Map all detection coordinates to a new origin
    param origin: (x,y) coordinates of the new origin
    param detections_txt: path to the detection txt file
    return: list of (x,y) coordinates in meters relative to new origin
    """
    

    

# print(find_point_projection((0, 0), 5280, 3956, 3, np.radians(-60)))
# print(find_point_projection((0, -3956/2), 5280, 3956, 3, np.radians(-60)))
# print(find_point_projection((0, 3956/2), 5280, 3956, 3, np.radians(-60)))