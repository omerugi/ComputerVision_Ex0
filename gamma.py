"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import ex1_utils as ex1
import numpy as np
from matplotlib import pyplot as plt
import cv2.cv2 as cv2

from ex1_utils import LOAD_GRAY_SCALE

def gamma(x):
    return x/10

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img = ex1.imReadAndConvert(img_path, rep)*255
    cv2.namedWindow('Gamma Currction')

    cv2.createTrackbar('Gamma', 'Gamma Currction', 1, 20, gamma)
    c = 1
    while 1 :
     #   cv2.imshow('Gamma Currction', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        b = cv2.getTrackbarPos('Gamma', 'Gamma Currction')
        img = c*np.power(img,b)



    pass


def main():
    gammaDisplay(('/home/omerugi/PycharmProjects/Ex0/bac_con.png', LOAD_GRAY_SCALE))


if __name__ == '__main__':
    main()
