import ultralytics
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import re


# find a way to multiprocess this

def divideImage(parent_directory, image_path, output_dir, img_dim=640, iou_thresh=0.5, conf_thresh=0.35):
	image_f = Image.open(image_path)
	image_f = cv2.cvtColor(np.array(image_f), cv2.COLOR_RGB2BGR)
	im = Image.open(image_path)
	img_type = str(im.format)
	if img_type == "MPO":
		img_type = "JPG"
	x, y = im.size

	#ceiling division
	num_of_row = -(x // -img_dim)
	x_padding = img_dim - (x // num_of_row)
	num_of_col = -(y // -img_dim)
	y_padding = img_dim - (y // num_of_col)
	for i in range(0, x, img_dim):
		for j in range(0, y, img_dim):
			#adds 10 pixel padding to the images
			start_i = (i - x_padding * (i / img_dim))
			start_j = (j - y_padding * (j / img_dim))
			low_x =  start_i / x
			low_y =  start_j/ y
			high_x = (start_i + img_dim) / x
			high_y = (start_j + img_dim) / y
			
			#overlap the last row/column images
			if (i + img_dim) > x:
				high_x = 1
				low_x = 1 - img_dim / x
				start_i = x - img_dim
			if (j + img_dim) > y:
				high_y = 1
				low_y = 1 - img_dim / y
				start_j = y - img_dim
				
			newIm = im.crop((start_i, start_j, start_i + img_dim, start_j + img_dim))
			
			result = model.predict(source = newIm, save=False, imgsz=img_dim, line_width=3, show_labels=False, show_conf=False, max_det = 3000, iou=iou_thresh, conf=conf_thresh)
			if (result[0].obb) != None:
				boxes = result[0].obb.xyxyxyxy.tolist()
			else:
				boxes = []
			w, h = x, y
			with open(parent_directory + output_dir + image[:-4] + ".txt", "a") as f:
				for box in boxes:
					line = f"0 {(box[0][0] + start_i)/w} {(box[0][1] + start_j)/h} {(box[1][0] + start_i)/w} {(box[1][1] + start_j)/h} {(box[2][0] + start_i)/w} {(box[2][1] + start_j)/h} {(box[3][0] + start_i)/w} {(box[3][1] + start_j)/h}"
					print(line, file = f)
					pts = [[box[0][0] + start_i,box[0][1] + start_j], [box[1][0] + start_i,box[1][1] + start_j], [box[2][0] + start_i,box[2][1] + start_j], [box[3][0] + start_i,box[3][1] + start_j]]
					cv2.polylines(image_f, np.int32([pts]), True, (0,0,255), 1)
	cv2.imwrite(parent_directory + output_dir + image, image_f)
	im.close()


# parent_directory = (ans + "/" if (ans := input("Enter full path to parent directory: ").strip()) != "-1" else "./")
# image_folder_dir = input("Enter relative path to image folder: ").strip() + "/"
# weight_path = (ans if (ans := input("Enter relative path to weight file: ").strip()) != "-1" else "best.pt")

# for timing purposes, hardcoding the paths for now
parent_directory = "./"
image_folder_dir = "DJI_202508081433_021_PineIslandbog5H3m5x3photo/"
weight_path = "best.pt"

model = ultralytics.YOLO(os.path.join(parent_directory, weight_path))
images = os.listdir(os.path.join(parent_directory, image_folder_dir))
images_list = []

for file in images:
	if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.mpo')):
		images_list.append(file)


# Logic to handle output_dir
# use regex to check if output dir exists, if exists, find max number and add 1 then create output{num} 
# if not, create output dir
pattern = re.compile(r"^output(\d*)$")
counter = 1
for directory in os.listdir(parent_directory):
	if os.path.isdir(os.path.join(parent_directory, directory)):
		match = pattern.match(directory)
		if match:
			counter = max(counter, int(match.group(1) if match.group(1) != '' else 0)) + 1
			
output_dir = f"output{counter}/" if counter > 1 else "output/"
print(f"Output will be saved in: {output_dir}\n")
os.mkdir(os.path.join(parent_directory, output_dir))





print("executing...")
for image in images_list:
	image_path = os.path.join(parent_directory, image_folder_dir, image)
	divideImage(parent_directory, image_path, output_dir, img_dim=640, iou_thresh=0.5, conf_thresh=0.35)