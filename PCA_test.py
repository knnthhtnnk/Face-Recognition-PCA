import os
import os.path
import cv2
import sklearn
import numpy as np 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
print("dafuq")
imageDir = "D:\\testDatabase" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg"] #specify your valid extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))

num = 1
img_list = []
img_arr = np.array([[]])


for idx,imagePath in enumerate(image_path_list):
    img = cv2.imread(imagePath)
    if img is None:
        continue
    print(type(img))
    img = img[0:248,0:248,0]
    img = img.flatten()    
    
    img = img.tolist()   
    # print(img) 
    img_list.append(img)
    
img_arr = np.array(img_list)
# print(img_list)
print((type(img_arr)))
print((type(img_arr[1])))
print((type(img_arr[1][1])))
# print(img_arr)
img_arr.astype('float64') 

pca = decomposition.PCA(n_components=20)
pca.fit(img_arr)
img_pcad = pca.transform(img_arr)
print(img_pcad)

test_img_dir = "D:\\Cropped Webcam Faces\\image_1.jpg"
test_img = cv2.imread(imagePath)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
print(test_img)
test_img = test_img[0:248,0:248]
test_img = test_img.flatten() 
test_img = pca.transform([test_img])
print(test_img)
distance_list = []
for img in img_pcad:
    distance = np.dot(test_img,img)
    distance_list.append(distance)    

print(distance_list)
print(image_path_list)