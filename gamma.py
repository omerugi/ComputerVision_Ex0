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

import numpy as np
from matplotlib import pyplot as plt
import cv2.cv2 as cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def gamma(x):
    return


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img = cv2.imread(img_path, rep - 1)
    cv2.namedWindow('Gamma correction')
    cv2.createTrackbar('Gamma', 'Gamma correction', 1, 200, gamma)
    img = np.asarray(img)/255

    cv2.imshow('Gamma correction', img)
    k = cv2.waitKey(1)

    newim = img

    while 1:
        cv2.imshow('Gamma correction', newim)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        g = cv2.getTrackbarPos('Gamma', 'Gamma correction')
        print(g/100)
        newim = np.power(img, g/100)

    cv2.destroyAllWindows()

    pass


def main():
    gammaDisplay('/home/omerugi/PycharmProjects/Ex0/beach.jpg', 2)


if __name__ == '__main__':
    main()
