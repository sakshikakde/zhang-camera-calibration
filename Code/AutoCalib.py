
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import math
import os
from Utils.MiscUtils import *
from Utils.ImageUtils import *
from Utils.MathUtils import *
import scipy.optimize 
import argparse
square_side = 12.5


def reprojectPointsAndGetError(A, kc, all_RT, all_image_corners, world_corners):

    error_mat = []
    alpha, gamma, beta, u0, v0, k1, k2 = extractParamFromA(A, kc)
    all_reprojected_points = []
    for i, image_corners in enumerate(all_image_corners): # for all images

        #RT for 3d world points
        RT = all_RT[i]
        #get ART for 2d world points
        RT3 = np.array([RT[:,0], RT[:,1], RT[:,3]]).reshape(3,3) #review
        RT3 = RT3.T
        ART3 = np.dot(A, RT3)

        image_total_error = 0
        reprojected_points = []
        for j in range(world_corners.shape[0]):

            world_point_2d = world_corners[j]
            world_point_2d_homo = np.array([world_point_2d[0], world_point_2d[1], 1]).reshape(3,1)
            world_point_3d_homo = np.array([world_point_2d[0], world_point_2d[1], 0, 1]).reshape(4,1)

            #get radius of distortion
            XYZ = np.dot(RT, world_point_3d_homo)
            x =  XYZ[0] / XYZ[2]
            y = XYZ[1] / XYZ[2]
            # x = alpha * XYZ[0] / XYZ[2] #assume gamma as 0 
            # y = beta * XYZ[1] / XYZ[2] #assume gamma as 0
            r = np.sqrt(x**2 + y**2) #radius of distortion

            #observed image co-ordinates
            mij = image_corners[j]
            mij = np.array([mij[0], mij[1], 1], dtype = 'float').reshape(3,1)

            #projected image co-ordinates
            uvw = np.dot(ART3, world_point_2d_homo)
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]

            u_dash = u + (u - u0) * (k1 * r**2 + k2 * r**4)
            v_dash = v + (v - v0) * (k1 * r**2 + k2 * r**4)
            reprojected_points.append([u_dash, v_dash])

            mij_dash = np.array([u_dash, v_dash, 1], dtype = 'float').reshape(3,1)

            #get error
            e = np.linalg.norm((mij - mij_dash), ord=2)
            image_total_error = image_total_error + e
        
        all_reprojected_points.append(reprojected_points)
        error_mat.append(image_total_error)
    error_mat = np.array(error_mat)
    error_average = np.sum(error_mat) / (len(all_image_corners) * world_corners.shape[0])
    # error_reprojection = np.sqrt(error_average)
    return error_average, all_reprojected_points


def lossFunc(x0, init_all_RT, all_image_corners, world_corners):

    A, kc = retrieveA(x0)
    alpha, gamma, beta, u0, v0, k1, k2 = x0

    error_mat = []

    for i, image_corners in enumerate(all_image_corners): # for all images

        #RT for 3d world points
        RT = init_all_RT[i]
        #get ART for 2d world points
        RT3 = np.array([RT[:,0], RT[:,1], RT[:,3]]).reshape(3,3) #review
        RT3 = RT3.T
        ART3 = np.dot(A, RT3)

        image_total_error = 0

        for j in range(world_corners.shape[0]):

            world_point_2d = world_corners[j]
            world_point_2d_homo = np.array([world_point_2d[0], world_point_2d[1], 1]).reshape(3,1)
            world_point_3d_homo = np.array([world_point_2d[0], world_point_2d[1], 0, 1]).reshape(4,1)

            #get radius of distortion
            XYZ = np.dot(RT, world_point_3d_homo)
            x =  XYZ[0] / XYZ[2]
            y = XYZ[1] / XYZ[2]
            # x = alpha * XYZ[0] / XYZ[2] #assume gamma as 0 
            # y = beta * XYZ[1] / XYZ[2] #assume gamma as 0
            r = np.sqrt(x**2 + y**2) #radius of distortion

            #observed image co-ordinates
            mij = image_corners[j]
            mij = np.array([mij[0], mij[1], 1], dtype = 'float').reshape(3,1)

            #projected image co-ordinates
            uvw = np.dot(ART3, world_point_2d_homo)
            u = uvw[0] / uvw[2]
            v = uvw[1] / uvw[2]

            u_dash = u + (u - u0) * (k1 * r**2 + k2 * r**4)
            v_dash = v + (v - v0) * (k1 * r**2 + k2 * r**4)

            mij_dash = np.array([u_dash, v_dash, 1], dtype = 'float').reshape(3,1)

            #get error
            e = np.linalg.norm((mij - mij_dash), ord=2)
            image_total_error = image_total_error + e

        error_mat.append(image_total_error / 54)
    
    return np.array(error_mat)
        


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImageFolderPath', default= './Data/Calibration_Imgs', help='Data path')
    Parser.add_argument('--SaveFolderePath', default='./Results/', help='Saved video file name')

    Args = Parser.parse_args()
    folder_name = Args.ImageFolderPath
    save_folder = Args.SaveFolderePath  

    images = loadImages(folder_name)
    h, w = [6,9]
    all_image_corners = getImagesPoints(images, h, w)
    world_corners = getWorldPoints(square_side, h, w)

    displayCorners(images, all_image_corners, h, w, save_folder)

    print("Calculating H for %d images", len(images))
    all_H_init = getAllH(all_image_corners, square_side, h, w)
    print("Calculating B")
    B_init = getB(all_H_init)
    print("Estimated B = ", B_init)
    print("Calculating A")
    A_init = getA(B_init)
    print("Initialized A = ",A_init)
    print("Calculating rotation and translation")
    all_RT_init = getRotationAndTrans(A_init, all_H_init)
    print("Init Kc")
    kc_init = np.array([0,0]).reshape(2,1)
    print("Initialized kc = ", kc_init)

    print("Optimizing ...")
    x0 = extractParamFromA(A_init, kc_init)
    res = scipy.optimize.least_squares(fun=lossFunc, x0=x0, method="lm", args=[all_RT_init, all_image_corners, world_corners])
    x1 = res.x
    AK = retrieveA(x1)
    A_new = AK[0]
    kc_new = AK[1]

    previous_error, _ = reprojectPointsAndGetError(A_init, kc_init, all_RT_init, all_image_corners, world_corners)
    att_RT_new = getRotationAndTrans(A_new, all_H_init)
    new_error, points = reprojectPointsAndGetError(A_new, kc_new, att_RT_new, all_image_corners, world_corners)

    print("The error befor optimization was ", previous_error)
    print("The error after optimization is ", new_error)
    print("The A matrix is: ", A_new)

    K = np.array(A_new, np.float32).reshape(3,3)
    D = np.array([kc_new[0],kc_new[1], 0, 0] , np.float32)
    for i,image_points in enumerate(points):
        image = cv2.undistort(images[i], K, D)
        for point in image_points:
            x = int(point[0])
            y = int(point[1])
            image = cv2.circle(image, (x, y), 5, (0, 0, 255), 3)
        # cv2.imshow('frame', image)
        filename = save_folder + str(i) + "reproj.png"
        cv2.imwrite(filename, image)
        # cv2.waitKey()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


