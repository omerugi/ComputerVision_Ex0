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
    Will track nar to change the gamma rate in an image
    The gamma will be set as:
    s=cr^Gamma
    The track bar will be [0-2] in rational numbers.
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    # Reads the image
    img = cv2.imread(img_path, rep - 1)

    # Create the window
    cv2.namedWindow('Gamma correction')
    # Create the taskbar
    cv2.createTrackbar('Gamma', 'Gamma correction', 1, 200, gamma)
    # Normalize the img
    img = np.asarray(img)/255

    # Shows the image
    cv2.imshow('Gamma correction', img)
    k = cv2.waitKey(1)

    newim = img

    # Infinite loop until you press esc
    while 1:
        cv2.imshow('Gamma correction', newim)
        # Set a key that will stop the loop
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # Get the number from the taskbar
        g = cv2.getTrackbarPos('Gamma', 'Gamma correction')
        # Calculate the new image
        newim = np.power(img, g/100)

    cv2.destroyAllWindows()

    pass


def main():
    gammaDisplay('/home/omerugi/PycharmProjects/Ex0/beach.jpg', 2)


if __name__ == '__main__':
    main()
