import os
import cv2
import numpy as np
class image_processing():
    def __init__(self,path,kernel=(5,5)):
        self.kernel = np.ones((5,5),np.uint8)
        self.image = cv2.imread(path,0)
        cv2.imshow("original",self.image)
        cv2.waitKey(0)
    def erosion(self):
        self.ero = cv2.erode(self.image,self.kernel,iterations=1)
        cv2.imshow("erosion",self.ero)
        cv2.waitKey(0)
        return self.ero
    def dilation(self):
        self.dil = cv2.dilate(self.image,self.kernel,iterations=1)
        cv2.imshow("dilation",self.dil)
        cv2.waitKey(0)
        return self.dil
    def opening(self):
        self.open = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, self.kernel)
        cv2.imshow("opening",self.open)
        cv2.waitKey(0)
        return self.open
    def closing(self):
        self.close = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, self.kernel)
        cv2.imshow("opening",self.close)
        cv2.waitKey(0)
        return self.close

op1 = image_processing("practicum\operations_1.jpg")
op1.erosion()

op2 = image_processing("practicum\operations_2.jpg")
op2.dilation()

op3= image_processing("practicum\operations_3.jpg")
op3.opening()

op4 = image_processing("practicum\operations_4.jpg")
op4.closing()






