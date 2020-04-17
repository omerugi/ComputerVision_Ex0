import numpy as np
import cv2
import os

path = os.getcwd() + '/Ex0/bac_con.png'

# Load an color image in grayscale
img = cv2.imread('/home/omerugi/PycharmProjects/Ex0/bac_con.png',2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()