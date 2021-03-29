import cv2
import numpy as np
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def getImagesHomoPoints(images):
    all_corners = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            corners2 = corners2.reshape(-1,2)
            # corners2 = np.hstack((corners2.reshape(-1,2), np.ones((corners2.shape[0], 1))))
            all_corners.append(corners2)
    return all_corners

def displayCorners(images, all_corners):
    for i, image in enumerate(images):
        corners = all_corners[i]
        corners = np.float32(corners.reshape(-1, 1, 2))
        cv2.drawChessboardCorners(image, (9, 6), corners, True)
        img = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
        cv2.imshow('frame', img)
        cv2.waitKey()

    cv2.destroyAllWindows()