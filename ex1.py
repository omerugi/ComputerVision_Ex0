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
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208386052


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Open the img file

    if representation == LOAD_RGB:
        image = cv2.imread(filename, 1)
        data = np.asarray(image, dtype=np.float32)
        data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    else:
        image = cv2.imread(filename, 0)
        data = np.asarray(image, dtype=np.float32)
    # Conver into np array

    # normalize
    # aaf = data - data.min()
    # aad = data.max() - data.min()
    # data = aaf/aad

    return data / 255

    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    data = imReadAndConvert(filename, representation)
    if representation == LOAD_RGB:
        plt.imshow(data)
        plt.show()
    else:
        plt.imshow(data, cmap='gray')
        plt.show()

    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape=imgRGB.shape
    return np.dot(imgRGB.reshape(-1,3), yiq_from_rgb.transpose()).reshape(OrigShape)

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)

    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    if len(imgOrig.shape) == 2:

        img = imgOrig*255

        histOrig, bins = np.histogram(img.flatten(), 256, [0, 255])

        cdf = histOrig.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0)

        imgEq = cdf[img.astype('uint8')]
        histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])



    else:
        img = transformRGB2YIQ(imgOrig)
        img[:, :, 0] = img[:, :, 0] * 255

        histOrig, bins = np.histogram(img[:, :, 0].flatten(), 256, [0, 255])
        cdf = histOrig.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        img[:, :, 0] = cdf[img[:, :, 0].astype('uint8')]
        histEq, bins2 = np.histogram(img[:, :, 0].flatten(), 256, [0, 256])
        img[:, :, 0] = img[:, :, 0] / 255
        imgEq = transformYIQ2RGB(img)



    return imgEq, histOrig, histEq

    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    if len(imOrig.shape) == 2:
        img = imOrig * 255

        histOrig, bins = np.histogram(img.flatten(), 256)
        bins = np.arange(0, 256)
        print(len(histOrig))

        z_arr = np.arange(nQuant + 1)  # with 0 and 255
        q_arr = np.arange(nQuant, dtype=np.float32)

        for i in range(0, len(z_arr)):  # init
            z_arr[i] = round((i / nQuant) * len(histOrig))

        img_list = []
        mse_list = []

        print(z_arr)
        for k in range(0, nIter):

            for i in range(0, nQuant):
                q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1] + 1], weights=histOrig[z_arr[i]:z_arr[i + 1] + 1])

            for j in range(1, nQuant):
                z_arr[j] = (q_arr[j - 1] + q_arr[j]) / 2

            newimg = img.copy()
            for i in range(1, nQuant + 1):
                newimg[(newimg >= z_arr[i - 1]) & (newimg < z_arr[i])] = q_arr[i - 1]

            img_list.append(newimg)
            mse = pow(np.power(img - newimg, 2).sum(), 0.5) / img.size
            mse_list.append(mse)


    else:

        img = transformRGB2YIQ(imOrig) * 255
        histOrig , bins = np.histogram(img[:, :, 0].flatten(), 256, [0, 256])

        z_arr = np.arange(nQuant + 1)  # with 0 and 255
        q_arr = np.arange(nQuant)

        for i in z_arr:  # init
            z_arr[i] = round((i / nQuant) * len(histOrig - 1))

        img_list = []
        mse_list = []

        for k in range(0, nIter):

            for i in range(0, nQuant):
                q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1),
                                      weights=histOrig[z_arr[i]:z_arr[i + 1]].reshape(1, -1))

            for j in range(1, nQuant):
                z_arr[j] = round((q_arr[j - 1] + q_arr[j]) / 2)

            newimg = img.copy()
            for i in range(1, nQuant + 1):
                newimg[:, :, 0][(newimg[:, :, 0] > z_arr[i - 1]) & (newimg[:, :, 0] < z_arr[i])] = q_arr[i - 1]

            newimg = transformYIQ2RGB(newimg) / 255
            img_list.append(newimg)
            mse=pow(np.power(imOrig-newimg,2).sum(),0.5)/imOrig.size
            mse_list.append(mse)


    return img_list, mse_list

    pass

