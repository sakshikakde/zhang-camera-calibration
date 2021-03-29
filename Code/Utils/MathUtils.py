import numpy as np
import cv2

def getWorldPoints(square_side):
    h, w = [6, 9]
    Yi, Xi = np.indices((h, w)) 
    lin_homg_pts = np.stack((Xi.ravel() * square_side, Yi.ravel() * square_side)).astype(int).T
    return lin_homg_pts

def getAllH(all_corners, square_side):
    set1 = getWorldPoints(square_side)
    all_H = []
    for corners in all_corners:
        set2 = corners
        H, _ = cv2.findHomography(set1, set2)
        all_H.append(H)
    return all_H