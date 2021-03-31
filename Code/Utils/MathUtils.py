import numpy as np
import cv2

def getWorldPoints(square_side):
    h, w = [6, 9]
    Yi, Xi = np.indices((h, w)) 
    lin_homg_pts = np.stack(((Xi.ravel() + 1) * square_side, (Yi.ravel() + 1) * square_side)).T
    return lin_homg_pts

def getAllH(all_corners, square_side):
    set1 = getWorldPoints(square_side)
    all_H = []
    for corners in all_corners:
        set2 = corners
        H, _ = cv2.findHomography(set1, set2, cv2.RANSAC, 5.0)
        all_H.append(H)
    return all_H

def getVij(hi, hj):
    Vij = np.array([ hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], hi[1]*hj[1], hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], hi[2]*hj[2] ])
    return Vij.T

def getV(all_H):
    v = []
    for H in all_H:
        h1 = H[:,0]
        h2 = H[:,1]

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        v.append(v12.T)
        v.append((v11 - v22).T)
    return np.array(v)

def arrangeB(b):
    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,0] = b[3]
    B[2,1] = b[4]
    B[2,2] = b[5]
    return B


def getB(all_H):
    v = getV(all_H)
    # vb = 0
    U, sigma, V = np.linalg.svd(v)
    b = V[-1, :]
    print("B matrix is ", b)
    B = arrangeB(b)  
    return B

def getA(B):
    v0 = (B[0,1] * B[0,2] - B[0,0] * B[1,2])/(B[0,0] * B[1,1] - B[0,1]**2)
    lamb = B[2,2] - (B[0,2]**2 + v0 * (B[0,1] * B[0,2] - B[0,0] * B[1,2]))/B[0,0]
    alpha = np.sqrt(lamb/B[0,0])
    beta = np.sqrt(lamb * (B[0,0]/(B[0,0] * B[1,1] - B[0,1]**2)))
    gamma = -(B[0,1] * alpha**2 * beta) / lamb 
    u0 = (gamma * v0 / beta) - (B[0,2] * alpha**2 / lamb)

    A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    return A
    