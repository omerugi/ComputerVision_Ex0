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
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import cv2.cv2 as cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 203024989


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Open the img file

    if representation == LOAD_RGB:
        img = cv2.imread(filename, 1)
        arr = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)

    else:
        img = cv2.imread(filename, 0)
        arr = np.asarray(img, dtype=np.float32)

    return arr / 255

    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == LOAD_RGB:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

    plt.show()

    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    transMatrix = np.array([[0.299, 0.587, 0.114],
                            [0.59590059, -0.27455667, -0.32134392],
                            [0.21153661, -0.52273617, 0.31119955]]).transpose()
    shape = imgRGB.shape
    return np.dot(imgRGB.reshape(-1, 3), transMatrix).reshape(shape)

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transMatrix = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                          [0.59590059, -0.27455667, -0.32134392],
                                          [0.21153661, -0.52273617, 0.31119955]])).transpose()
    shape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), transMatrix).reshape(shape)
    pass


def hist_eq(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function will do histogram equalization on a given 1D np.array
    meaning will balance the colors in the image.
    For more details:
    https://en.wikipedia.org/wiki/Histogram_equalization
    **Original function was taken from open.cv**
    :param img: a 1D np.array that represent the image
    :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
    """

    # Flattning the image and converting it into a histogram
    histOrig, bins = np.histogram(img.flatten(), 256, [0, 255])
    # Calculating the cumsum of the histogram
    cdf = histOrig.cumsum()
    # Places where cdf = 0 is ignored and the rest is stored
    # in cdf_m
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Normalizing the cdf
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Filling it back with zeros
    cdf = np.ma.filled(cdf_m, 0)


    # Creating the new image based on the new cdf
    imgEq = cdf[img.astype('uint8')]
    histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    return imgEq, histOrig, histEq

    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        The function will fist check if the image is RGB or gray scale
        If the image is gray scale will equalizes
        If RGB will first convert to YIQ then equalizes the Y level
        :param imgOrig: Original Histogram
        :return: imgnew -> image after equalization, hist-> original histogram, histnew -> new histogram
    """

    if len(imgOrig.shape) == 2:

        img = imgOrig * 255
        imgEq, histOrig, histEq = hist_eq(img)

    else:
        img = transformRGB2YIQ(imgOrig)
        img[:, :, 0] = img[:, :, 0] * 255
        img[:, :, 0], histOrig, histEq = hist_eq(img[:, :, 0])
        img[:, :, 0] = img[:, :, 0] / 255
        imgEq = transformYIQ2RGB(img)

    return imgEq, histOrig, histEq

    pass


def init_hist_limits(img: np.ndarray, nQuant: int):
    """
    Creating the histogram of a given 1D image
    And given nQuant initializing an array size of nQuant+1
    That will be the limits that will devied the histogram in
    to equal parts.
    :param imOrig: 1D image
    :param nQuant: The number of pixels we wish to represent the new image
    :return: hist -> histogram of the image, z-> an array of the limits
    """
    hist, bins = np.histogram(img.flatten(), 256)
    z = np.arange(nQuant + 1)  # with 0 and 255

    for i in range(0, len(z)):  # init
        z[i] = round((i / nQuant) * len(hist))

    return z, hist
    pass


def calc_wg_avg_limits(z: np.ndarray, hist: np.ndarray, nQuant: int) -> np.array:
    """
       Quantize a single time a given histogram of an image.
       Calculating the weighted average (q) of each section of the histogram from z[i] to z[i+1]
       then placing the new limits in the average between q[i] and q[i+1]
       :param img: Original 1D image
       :param hist: the histogram we wish to Quantize
       :param nQuant: The number of color scale we wish to Quantize the image
       :param z: The limits of the histogram
       :return: the new z and q
       """
    # Init the array for the weighted averages
    q = np.arange(nQuant, dtype=np.float32)
    index = np.arange(0, 256)

    # Calculate the weighted average of each section
    for i in range(0, nQuant):
        # The function calculate the weighted average given an array-like and weights of each place in the array
        q[i] = np.average(index[z[i]:z[i + 1] + 1], weights=hist[z[i]:z[i + 1] + 1])

    # Calculating the new limits in the histogram
    for j in range(1, nQuant):
        z[j] = (q[j - 1] + q[j]) / 2

    return z, q
    pass

def replace_to_n_quant(img: np.ndarray,z: np.ndarray, q: np.ndarray, nQuant: int)->np.ndarray:
    """
          reconstructing the new image after the quantization
          :param img: Original 1D image
          :param z: The limits of the histogram
          :param q:the weighted average of each section
          :param nQuant: The number of color scale we wish to Quantize the image
          :return: newimg -> The new image after Quantize
          """
    newimg = img.copy()
    for i in range(0, nQuant):
        newimg[(newimg >= z[i]) & (newimg < z[i+1])] = q[i]
    return newimg

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
         The function will fist check if the image is RGB or gray scale
        If the image is gray scale will quantize
        If RGB will first convert to YIQ then quantized the Y level
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    # Init array of quantized images and mse
    imgs = []
    mses = []

    # Check color map
    if len(imOrig.shape) == 2:
        # "Anti-normalizing" to be in [0:255] range
        img = imOrig * 255
        # Get the histogram of the image and the first limits
        z, hist = init_hist_limits(img, nQuant)

    else:
        # Convering the RGB to YIQ
        # "Anti-normalizing" to be in [0:255] range
        img = transformRGB2YIQ(imOrig)*255
        # Get the histogram of the image and the first limits
        z, hist = init_hist_limits(img[:, :, 0], nQuant)

    for k in range(0, nIter):# Quantized the image nIter times
        z, q = calc_wg_avg_limits(z, hist, nQuant)

        if len(imOrig.shape) == 2:
            # Applying the weighted average of each section to the new image
            newimg=replace_to_n_quant(img, z, q, nQuant) / 255

        else:
            newimg=img.copy()
            # Applying the weighted average of each section to the new image
            newimg[:, :, 0]=replace_to_n_quant(newimg[:, :, 0], z, q, nQuant)
            # Normalizing and converting back to RGB
            newimg=transformYIQ2RGB(newimg/ 255)

        imgs.append(newimg)
        mse = np.power(imOrig-newimg,2).sum()/imOrig.size
        mses.append(mse)

    return imgs, mses

    pass
