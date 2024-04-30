#import basic module
import os
import time
from ximea import xiapi
import cv2
import time
import numpy as np
import csv
import pandas as pd
from scipy import linalg
from scipy.spatial.transform import Rotation
import scipy
import glob
import math

import triangulation as tri
import dlt


#3d positioning


def getCorner(cameraMatrix,dist,src,rootDir,checkerBoard,checkersize,debug = True,boolReverse=False) -> list:
    """get axis-end points from chessboard 
    Args:
        cameraMatrix (np.array): camera matrix
        dist (np.array): distortion coefficient
        src (char): image path
        checkerBoard (tuple): checker board size Defaults to CHECKERBOARD.
    Returns:
        tuple: original corners position
    """
    # setting 
    print(src)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    objp = np.zeros((checkerBoard[0]*checkerBoard[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerBoard[0],0:checkerBoard[1]].T.reshape(-1,2)
    objp = checkersize * objp
    # 3d axis setting : (x,y,z)
    axis = np.float32([[checkersize,0,0], [0,checkersize,0], [0,0,checkersize]]).reshape(-1,3)
    #print("index:",index)
    img = cv2.imread(src)
    if (len(img.shape) == 3):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret, corners = cv2.findChessboardCorners(gray, checkerBoard,None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
        if boolReverse:
            corners2 = corners2[::-1]
        original = list(i for i in corners2[0].ravel())
        if debug:
            cv2.circle(gray,tuple(int(i) for i in corners2[0].ravel()),radius=7,color=(255,0,0),thickness=-1)
            index = int((src.split("\\")[-1]).split(".")[0])
            saveDir = os.path.join(rootDir,"corners")
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            cv2.imwrite(os.path.join(saveDir,f"{index:02d}.png"),gray)
    else:
        print("can't find corners")
        original = [0,0]

    return original

def triangulate(left,right,K_left,K_right,dist_left,dist_right,R0,T0,R1,T1,baseline,method):
    result = []
    xl = left[0]
    yl = left[1]
    xr = right[0]
    yr = right[1]
    if method == "triangulation-mean":
        #calculate 3d position
        euclid_points = tri.reconstruction_mean((xl,yl),(xr,yr),K_left,K_right,dist_left,dist_right,baseline=baseline)
    elif method == "triangulation":
        euclid_points = tri.reconstruction((xl,yl),(xr,yr),K_left,K_right,dist_left,dist_right,baseline=baseline)
    elif method == "dlt" or "library":
        #param setting
        cam_left_param = [K_left, dist_left, R0, T0]
        cam_right_param = [K_right, dist_right, R1, T1]
        # make instance of DLTs
        instance_dlt = dlt.DLT(cam_left_param=cam_left_param, cam_right_param=cam_right_param)
        #undistort points
        distorted_left = np.array([[[xl,yl]]]).astype(float)
        distorted_right = np.array([[[xr,yr]]]).astype(float)
        undistorted_left = cv2.undistortPoints(distorted_left,K_left,dist_left,P=K_left)[0] #(1,2)
        undistorted_right = cv2.undistortPoints(distorted_right,K_right,dist_right,P=K_right)[0] #(1,2)
        # Reshape the undistorted points back to (N, 2) if needed
        undistorted_left = undistorted_left.reshape((-1, 2))
        undistorted_right = undistorted_right.reshape((-1, 2))
        for left, right in zip(undistorted_left,undistorted_right):
            #rint(left)
            if method == "dlt":
                euclid_points = instance_dlt.main(point1=left, point2=right)
            elif method=="library":
                pt3d = cv2.triangulatePoints(instance_dlt.p_left, instance_dlt.p_right, left, right)
                pt3d = pt3d[:3] / pt3d[3]
                euclid_points = pt3d.transpose()[0]
                #homog_points = pt3d.transpose()
                #euclid_points = cv2.convertPointsFromHomogeneous(homog_points)[0] #(1,1,3) -> (3,)
    print("3d position = ",euclid_points)
    result.append(euclid_points[0])
    result.append(euclid_points[1])
    result.append(euclid_points[2])
    return np.array(result)

def rmse(pos_gt,pos_est):
    rmse_pos = 0
    rmse_x = 0
    rmse_y = 0
    rmse_z = 0
    rmse = 0
    for i in range(pos_gt.shape[0]):
        for j in range(3):
            rmse_pos += (pos_gt[i][j]-pos_est[i][j])**2
            if j == 0:
                rmse_x = abs(pos_gt[i][j]-pos_est[i][j])
            elif j == 1:
                rmse_y = abs(pos_gt[i][j]-pos_est[i][j])
            elif j == 2:
                rmse_z = abs(pos_gt[i][j]-pos_est[i][j])
        rmse_pos = round((rmse_pos)**(1/2),2)
        print(f"{i} :: rmse :: position = {rmse_pos} [mm], x = {rmse_x} [mm], y= {rmse_y} [mm], z = {rmse_z} [mm]")
        rmse+= rmse_pos
    rmse /= pos_gt.shape[0] 
    print(f"(total) rmse :: ave = {rmse_pos} [mm]")