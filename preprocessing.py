import numpy as np
import cv2
import os
import csv
from image_preprocessing import func

# Ensure necessary directories exist
# if not os.path.exists("data2"):
#     os.makedirs("data2")
# if not os.path.exists("data2/train"):
#     os.makedirs("data2/train")
# if not os.path.exists("data2/test"):
#     os.makedirs("data2/test")

# if not os.path.exists("data2/test/blank"):
#     os.makedirs("data2/test/blank") 

# if not os.path.exists("data2/train/blank"):
#     os.makedirs("data2/train/blank") 

if not os.path.exists("data4"):
    os.makedirs("data4")
if not os.path.exists("data4/train"):
    os.makedirs("data4/train")
if not os.path.exists("data4/test"):
    os.makedirs("data4/test")

if not os.path.exists("data4/test/blank"):
    os.makedirs("data4/test/blank") 

if not os.path.exists("data4/train/blank"):
    os.makedirs("data4/train/blank") 

# Paths
# path = "data/train"
# path = "data/test"
# path = "data3/train"
path = "data3/test"

# path1 = "data2"
path1 = "data4"

# CSV header
a = ['label']
for i in range(64 * 64):
    a.append("pixel" + str(i))

# Initialize variables
label = 0
var = 0
c1 = 0
c2 = 0

# Walk through the directories
for (dirpath, dirnames, filenames) in os.walk(path):
    # print(f"Current directory: {dirpath}")
    # print(f"Subdirectories: {dirnames}")
    # print(f"Files: {filenames}")
    for dirname in dirnames:
        # print(f"Processing directory: {dirname}")
        for (direcpath, direcnames, files) in os.walk(os.path.join(path, dirname)):
            # print(f"Inside directory: {direcpath}")
            # print(f"Files in directory: {files}")
            if not os.path.exists(os.path.join(path1, "train", dirname)):
                os.makedirs(os.path.join(path1, "train", dirname))
            if not os.path.exists(os.path.join(path1, "test", dirname)):
                os.makedirs(os.path.join(path1, "test", dirname))
                
            num = 100000000000000000
            i = 100000000000000001
            for file in files:
                var += 1
                actual_path = os.path.join(path, dirname, file)
                actual_path1 = os.path.join(path1, "train", dirname, file)
                actual_path2 = os.path.join(path1, "test", dirname, file)
                img = cv2.imread(actual_path, 0)
                bw_image = func(actual_path)
                if i < num:
                    c1 += 1
                    cv2.imwrite(actual_path1, bw_image)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2, bw_image)
                i += 1
        label += 1

print(var)
print(c1)
print(c2)
