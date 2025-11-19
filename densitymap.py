import georef2
import numpy as np
import pyexiv2
import os

# populate map
ORIGIN_PATH = "DJI_202508081433_021_PineIslandbog5H3m5x3photo/DJI_20250808143604_0001_D_Waypoint1.JPG"
FORWARD_DIR_PATH = "DJI_202508081433_021_PineIslandbog5H3m5x3photo/DJI_20250808143611_0002_D_Waypoint2.JPG"
IMG_DIR = "DJI_202508081433_021_PineIslandbog5H3m5x3photo"
LABEL_DIR = "output"
all_detections_coor = []

path_list = sorted(os.listdir("output"))
label_list = []
img_list = []

for i in range(len(path_list)): 
    if path_list[i][-3:] == 'txt':
        label_list.append(path_list[i])
    elif path_list[i][-3:] == 'JPG':
        img_list.append(path_list[i])
    else:
        pass

for i in range(len(label_list)):
    img_path = os.path.join(IMG_DIR, img_list[i])
    label_path = os.path.join(LABEL_DIR, label_list[i])
    mapped_list = georef2.georef(ORIGIN_PATH, FORWARD_DIR_PATH, img_path, label_path)
    all_detections_coor.extend(mapped_list)

with open("./all_detections.txt", "w") as f:
    for coor in all_detections_coor:
        to_write = str(coor[0]) + ", " + str(coor[1]) + "\n"
        f.write(to_write)
    f.close()

# finished mapping detections to relative coordinate with drone's first image as basis