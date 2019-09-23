import cv2
import numpy as np
import os

imread = cv2.imread("fingers/train/0a1a077a-5197-44d9-8607-248bf244930d_3R.png", cv2.IMREAD_GRAYSCALE)
print(imread.shape)
cv2.imshow("output", imread)
cv2.waitKey(0)