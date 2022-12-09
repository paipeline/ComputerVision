import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread("practicum\shanghai.jpeg",cv.IMREAD_GRAYSCALE)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv.imshow("img",img)
plt.hist(img.ravel(),256,[0,256])

cv.waitKey(0)
cv.destroyAllWindows()
plt.show()
