import numpy as np
import math
import os
from PIL import Image
from PIL import ExifTags
import defusedxml
import shutil


def find_y(theta_y, pitch_angle, height):
    theta = 90 - pitch_angle + theta_y
    theta = math.pi * theta / 180
    AB = height * math.tan(theta)
    OB = height / math.cos(theta)
    return AB, OB


def find_angles(height, angle_y, x_coord, y_coord):
    angle_y = math.pi * angle_y / 180
    DO = height / 2 / math.tan(angle_y / 2)
    BO = math.sqrt(DO * DO + y_coord * y_coord)
    theta_x = math.atan(x_coord / BO) * 180 / math.pi
    # print(np.radians(theta_x))
    theta_y = math.atan(y_coord / DO) * 180 / math.pi
    ###Calculate verticle angle of view and print###
    temp = 5280 / 2 / DO
    # print(math.atan(temp))
    ###end###
    return theta_x, theta_y


def find_x(OB, theta_x):
    theta_x = math.pi * theta_x / 180
    return OB * math.tan(theta_x)


def find_x_y(image_width, image_height, verticle_AOV, x_coord, y_coord, pitch_angle, height):
    x_coord = x_coord - image_width / 2
    y_coord = y_coord - image_height / 2
    theta_x, theta_y = find_angles(image_height, verticle_AOV, x_coord, y_coord)
    y, OB = find_y(theta_y, pitch_angle, height)
    x = find_x(OB, theta_x)
    return x, y


def to_decimal_degrees(d, m, s):
    return d + (m / 60.0) + (s / 3600.0)


def gpsinfo(img):
    exif_data = img._getexif()
    if exif_data is None:
        print('Sorry, image has no exif data.')
        exit()
    else:
        for key, val in exif_data.items():
            if ExifTags.TAGS[key] == "GPSInfo":
                return val


def image_direction(img):
    yaw = 0
    pitch = 0
    drone_height = 0
    xmp_data = img.getxmp()
    if xmp_data is None:
        print('Sorry, image has no exif data.')
        exit()
    else:
        xmp_data = xmp_data["xmpmeta"]["RDF"]["Description"]
        pitch = xmp_data["GimbalPitchDegree"]
        yaw = xmp_data["GimbalYawDegree"]
        drone_height = xmp_data["RelativeAltitude"]
        if drone_height[0] == "+":
            drone_height = float(drone_height[1:])
        else:
            drone_height = -1.0 * float(drone_height[1:])

    return float(yaw), float(pitch), drone_height


# keys from https://pillow.readthedocs.io/en/stable/_modules/PIL/ExifTags.html#GPS
def getGPS(path):
    gpsin = gpsinfo(path)
    latitude = to_decimal_degrees(gpsin[2][0], gpsin[2][1], gpsin[2][2])
    longitude = to_decimal_degrees(gpsin[4][0], gpsin[4][1], gpsin[4][2])
    altitude = gpsin[6]
    if gpsin[5] == b'\x01':
        altitude = -1.0 * altitude
    if gpsin[3] == 'W':
        longitude = -1.0 * longitude
    if gpsin[1] == 'S':
        latitude = -1.0 * latitude
    return latitude, longitude, altitude


def leaf_gps(rel_x, rel_y, lat, lon, altitude):
    r = 6371000 + altitude
    latitude = lat + (2 * math.asin((rel_y / (2 * r))) * 180) / math.pi
    longitude = lon + (2 * math.asin((rel_x / (2 * r))) * 180) / math.pi
    return latitude, longitude


#   Parameters: image path and label path
#   return:     list of gps coordinates in the format [[lat,long],[lat,long]]
def georef(img_path, label_path):
    gps_list = []
    if os.path.exists(label_path):
        img = Image.open(img_path)
        latitude, longitude, altitude = getGPS(img)
        width, height = img.size
        print("width and heights are: ", width, height)
        yaw, pitch, drone_height = image_direction(img)
        print("yaw, pitch: ", yaw, pitch)
        print("gps: ", latitude, longitude, altitude)
        with open(label_path) as file:
            lines = file.read().splitlines()
        for line in lines:
            if (line.strip() != ""):
                # oriented bounding box image coordinates
                boxlist = list(line.split(" "))
                box = [(float(boxlist[1]), float(boxlist[2])), (float(boxlist[3]), float(boxlist[4])),
                       (float(boxlist[5]), float(boxlist[6])), (float(boxlist[7]), float(boxlist[8]))]

                # Only find the gps coordinates of the 1st corner
                # image coordinates to real life coordinates (relative distance from the drone/camera)
                x, y = find_x_y(width, height, 55.07, box[0][0] * width, box[0][1] * height, -1.0 * pitch, drone_height)
                # gps_list.append([x, y])

                # Calculate x, y where y-axis is noth.
                theta = -1.0 * yaw * math.pi / 180
                x = x * math.cos(theta) + y * math.sin(theta)
                y = -1.0 * x * math.sin(theta) + y * math.cos(theta)

                lat, long = leaf_gps(x, y, latitude, longitude, altitude)
                gps_list.append([lat, long])
    return gps_list

# print("compile successfully")

# gps_list = georef("DJI_202508081433_021_PineIslandbog5H3m5x3photo/DJI_20250808143611_0002_D_Waypoint2.JPG", "output/DJI_20250808143611_0002_D_Waypoint2.txt")
# # print(gps_list)
# with open("./gps_output.txt", "w") as f:
#     for gps in gps_list:
#         to_write = str(gps[0]) + ", " + str(gps[1]) + "\n"
#         f.write(to_write)

# print(find_x_y(5280, 3956, 55.07, 5280/2, 0, 60, 3))