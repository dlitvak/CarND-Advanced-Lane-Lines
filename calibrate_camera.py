import cv2 as cv
import numpy as np
import glob
import matplotlib.image as mpimg

obj_pnts = []
img_pnts = []

objp = np.zeros((6*9, 3), dtype=np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

imgs = glob.glob("camera_cal/calibration*.jpg")
imgShape = None

for fname in imgs:
    img = mpimg.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    if ret:
        img_pnts.append(corners)
        obj_pnts.append(objp)
        imgShape = gray.shape[::-1]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_pnts, img_pnts, imgShape, None, None)

np.savez("camera_calib_result.npz", mtx=mtx, dist=dist)

