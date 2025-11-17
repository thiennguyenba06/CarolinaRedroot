# Ba Thien Nguyen Notes
# fix minor error in path assignment
# for directory path copy absolute path and add / at the end
# for image folder name copy relative path and add / at the end
# for weight path copy relative path
# image folder and weight path are then concatenated to the directory path
# .txt files contain relative coordinates of boxes rounding detected objects in the images


# parameters
# angle at which the camera is tilted is "GimbalPitchDegree" in image xmp data

import ultralytics
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from pathlib import Path



def divideImage(parent_directory, image_path, img_dim=640, iou_thresh=0.5, conf_thresh=0.35):
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
      with open(parent_directory + "output/" + image[:-4] + ".txt", "a") as f:
        for box in boxes:
          line = f"0 {(box[0][0] + start_i)/w} {(box[0][1] + start_j)/h} {(box[1][0] + start_i)/w} {(box[1][1] + start_j)/h} {(box[2][0] + start_i)/w} {(box[2][1] + start_j)/h} {(box[3][0] + start_i)/w} {(box[3][1] + start_j)/h}"
          print(line, file = f)
          pts = [[box[0][0] + start_i,box[0][1] + start_j], [box[1][0] + start_i,box[1][1] + start_j], [box[2][0] + start_i,box[2][1] + start_j], [box[3][0] + start_i,box[3][1] + start_j]]
          cv2.polylines(image_f, np.int32([pts]), True, (0,0,255), 1)
  cv2.imwrite(parent_directory + "output/" + image, image_f)

  im.close()




dataset_path = "./" # current directory
weight_path = "best.pt" # params for model
image_folder = "DJI_202508081433_021_PineIslandbog5H3m5x3photo/"

print("split_predict script")

# print("Enter the iou threshhold to use [ex) 0.5]: ", end=" ")
# # iou_thresh = float(input())
iou_thresh = 0.5

# print("Enter the confidence threshhold to use [ex) 0.35]: ", end=" ")
# # conf_thresh = float(input())
conf_thresh = 0.35

#Change directory to dataset directory
parent_directory = dataset_path
print("parent directory is: ", parent_directory)

model = ultralytics.YOLO(parent_directory + weight_path) 
print("images path: " + parent_directory + image_folder)
images = os.listdir(parent_directory + image_folder)
images_list = []

for image_f in images:
  print(image_f)
  if image_f == ".ipynb_checkpoints":
    continue
  images_list.append(image_f)

if(os.path.exists(parent_directory + "output/")):
  #shutil.rmtree(parent_directory + "output/")
  print("output file already exists. Using the same output folder")
else:
    os.mkdir(parent_directory + "output/")

for image in images_list:
  image_path = parent_directory + image_folder + image
  if (float(Image.open(image_path).getxmp()['xmpmeta']['RDF']['Description']['GimbalPitchDegree']) > -50):
    print("skipped image: " + image)
    continue
  divideImage(parent_directory, image_path, 640, iou_thresh, conf_thresh)