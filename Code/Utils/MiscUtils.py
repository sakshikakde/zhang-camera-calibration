import cv2
import os
import numpy as np

def loadImages(folder_name):
    files = os.listdir(folder_name)
    print("Loading images from ", folder_name)
    images = []
    for f in files:
        # print(f)
        image_path = folder_name + "/" + f
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def extractParamFromA(init_A, init_kc):
    alpha = init_A[0,0]
    gamma = init_A[0,1]
    beta = init_A[1,1]
    u0 = init_A[0,2]
    v0 = init_A[1,2]
    k1 = init_kc[0]
    k2 = init_kc[1]

    x0 = np.array([alpha, gamma, beta, u0, v0, k1, k2])
    return x0

def retrieveA(x0):
    alpha, gamma, beta, u0, v0, k1, k2 = x0
    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]]).reshape(3,3)
    kc = np.array([k1, k2]).reshape(2,1)
    return A, kc