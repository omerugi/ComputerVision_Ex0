

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
from sklearn.preprocessing import Normalizer
from math import pow


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
        data = ((data - data.min()) / data.max() - data.min())
        #data = data / (pow(np.power(data,2).sum(), 0.5))

    else:
        image = cv2.imread(filename, 0)
        data = np.asarray(image, dtype=np.float32)



    print(data)
    return data

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

    YIQ = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb).reshape(imgRGB.shape)
    print(YIQ[:, :, 0])
    plt.imshow(YIQ)
    plt.show()
    return YIQ

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
    rgb_from_yiq = np.linalg.inv(yiq_from_rgb)
    RGB = np.dot(imgYIQ.reshape(-1, 3), rgb_from_yiq).reshape(imgYIQ.shape)


    return RGB/255
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    if len(imgOrig.shape)==2:
        img=imgOrig*255
        histOrig, bins = np.histogram(img.flatten(), 256, [0, 256])

        cdf = histOrig.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        imgEq = cdf[img.astype('uint8')]
        histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])

    else:
        img = transformRGB2YIQ(imgOrig)*255
        histOrig, bins = np.histogram(img[:, :, 0].flatten(), 256, [0, 256])

        cdf = histOrig.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        img[:, :, 0] = cdf[img[:, :, 0].astype('uint8')]
        histEq, bins2 = np.histogram(img[:, :, 0].flatten(), 256, [0, 256])

        imgEq = transformYIQ2RGB(img)


    plt.imshow(imgEq, cmap='gray')
    plt.show()

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
        img = imOrig*255
        histOrig, bins = np.histogram(img.flatten(), 256, [0, 256])

        z_arr = np.arange(nQuant+1)# with 0 and 255
        q_arr = np.arange(nQuant)

        for i in range(0, len(z_arr)):#init
            z_arr[i] = round((i/nQuant) * len(histOrig-1))

        img_list = []
        mse_list = []

        for k in range(0, nIter):

            for i in range(0, nQuant):
                q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1), weights=histOrig[z_arr[i]:z_arr[i + 1]].reshape(1, -1))
                print(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1))
                print(z_arr[i+1])

            for j in range(1, nQuant):
                z_arr[j] = round((q_arr[j-1]+q_arr[j])/2)

            newimg = img
            for i in range(1, nQuant + 1):
                newimg[(newimg >= z_arr[i - 1]) & (newimg < z_arr[i])] = q_arr[i - 1]

            img_list.append(newimg)
            mse_list.append(mean_squared_error(img, newimg))

        newimg = img
        for i in range(1, nQuant + 1):
            newimg[(newimg > z_arr[i - 1]) & (newimg < z_arr[i])] = q_arr[i - 1]

        plt.imshow(newimg, cmap='gray')
        plt.show()
        if len(imOrig.shape) == 2:
            img = imOrig * 255
            histOrig, bins = np.histogram(img.flatten(), 256, [0, 256])

            z_arr = np.arange(nQuant + 1)  # with 0 and 255
            q_arr = np.arange(nQuant)

            for i in z_arr:  # init
                z_arr[i] = round((i / nQuant) * len(histOrig - 1))

            img_list = [0] * nIter
            mse_list = [0] * nIter

            for k in range(0, nIter):

                for i in range(0, nQuant):
                    q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1),
                                          weights=histOrig[z_arr[i]:z_arr[i + 1]].reshape(1, -1))
                    print(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1))
                    print(z_arr[i + 1])

                for j in range(1, nQuant):
                    z_arr[j] = round((q_arr[j - 1] + q_arr[j]) / 2)

                newimg = img
                for i in range(1, nQuant + 1):
                    newimg[(newimg >= z_arr[i - 1]) & (newimg < z_arr[i])] = q_arr[i - 1]

                plt.imshow(newimg, cmap='gray')
                img_list.append(newimg/255)
                mse_list.append(mean_squared_error(img/255, newimg/255))


            plt.show()

    else:
        img = transformRGB2YIQ(imOrig)*255
        histOrig, bins = np.histogram(img[:, :, 0].flatten(), 256, [0, 256])

        z_arr = np.arange(nQuant+1)# with 0 and 255
        q_arr = np.arange(nQuant)

        for i in z_arr:#init
            z_arr[i] = round((i/nQuant) * len(histOrig-1))

        img_list=[]
        mse_list=[]

        for k in range(0, nIter):

            for i in range(0, nQuant):
                q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1), weights=histOrig[z_arr[i]:z_arr[i + 1]].reshape(1, -1))
                print(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1))
                print(z_arr[i+1])

            for j in range(1, nQuant):
                z_arr[j] = round((q_arr[j-1]+q_arr[j])/2)

            for i in range(1, nQuant + 1):
                img[:, :, 0][(img[:, :, 0] > z_arr[i - 1]) & (img[:, :, 0] < z_arr[i])] = q_arr[i - 1]


            newimg = transformYIQ2RGB(img)/255

            plt.imshow(newimg)
            plt.show()

            img_list.append(newimg)
            mse_list.append(mean_squared_error(imOrig.reshape(1, -1), newimg.reshape(1, -1)))



    return img_list, mse_list
pass


imDisplay('/home/omerugi/PycharmProjects/Ex0/beach.jpg',2)

#transformRGB2YIQ(imReadAndConvert('/home/omerugi/PycharmProjects/Ex0/beach.jpg', 2))

#transformYIQ2RGB(transformRGB2YIQ(imReadAndConvert('/home/omerugi/PycharmProjects/Ex0/beach.jpg', 2)))

#hsitogramEqualize(imReadAndConvert('/home/omerugi/PycharmProjects/Ex0/beach.jpg', 2))
#hsitogramEqualize(imReadAndConvert('/home/omerugi/PycharmProjects/Ex0/bac_con.png', 1))

#quantizeImage(imReadAndConvert('/home/omerugi/PycharmProjects/Ex0/beach.jpg', 1),4,5)


exit(0)

