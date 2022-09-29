import cv2
import os
from glob import glob

png = glob("C:/Users/zuhal/Desktop/termal_drone/yolov4/obj/*.png")
for j in png:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3]+"jpg",img)
    os.remove(j)