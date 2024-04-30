# import function
import stereo2

# import basic module
import os
import time
from ximea import xiapi
import cv2
import time
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.transform import Rotation
import glob
import matplotlib.pyplot as plt


BASELINE = 280


def reconstruction(
    point1, point2, CameraMatrix_left, CameraMatrix_right,dist_left,dist_right, baseline=BASELINE
) -> list:
    """calculate 3d position
    Args:
      point1(list):2d position of left camera [t,x,y]
      point2(list):2d position of right camera [t,x,y]
    Return:
      (list):3d position
    """
    f_x = CameraMatrix_left[0][0]
    f_y = CameraMatrix_left[1][1]
    f_skew = CameraMatrix_left[0][1]
    o_x = CameraMatrix_left[0][2]
    o_y = CameraMatrix_left[1][2]
    #(x_1, y_1) = point1
    #(x_2, y_2) = point2
    #undistort points
    left = np.array([[point1[0],point1[1]]]).reshape((1,-1,2))
    right = np.array([[point2[0],point2[1]]]).reshape((1,-1,2))
    undistorted_left = cv2.undistortPoints(left,CameraMatrix_left,dist_left,P=CameraMatrix_left)
    undistorted_right = cv2.undistortPoints(right,CameraMatrix_right,dist_right,P=CameraMatrix_right)
    undistorted_left = undistorted_left.reshape((-1, 2))
    undistorted_right = undistorted_right.reshape((-1, 2))
    [x_1,y_1] = undistorted_left[0]
    [x_2,y_2] = undistorted_right[0]
    disparity = x_1 - x_2
    X = round((baseline / disparity) * (x_1 - o_x - (f_skew / f_y) * (y_1 - o_y)), 2)
    Y = round(baseline * (f_x / f_y) * (y_1 - o_y) / disparity, 2)
    Z = round((f_x * baseline) / disparity, 2)
    return [X, Y, Z]


def reconstruction_mean(
    point1,
    point2,
    CameraMatrix1,
    CameraMatrix2,
    dist1,
    dist2,
    baseline=BASELINE,
) -> list:
    """calculate 3d position
    Args:
      point1(list):2d position of left camera [t,x,y]
      point2(list):2d position of right camera [t,x,y]
    Return:
      (list):3d position
    """
    f_x = (CameraMatrix1[0][0] + CameraMatrix2[0][0]) / 2
    f_y = (CameraMatrix1[1][1] + CameraMatrix2[1][1]) / 2
    f_skew = (CameraMatrix1[0][1] + CameraMatrix2[0][1]) / 2
    o_x = (CameraMatrix1[0][2] + CameraMatrix2[0][2]) / 2
    o_y = (CameraMatrix1[1][2] + CameraMatrix2[1][2]) / 2
    #(x_1, y_1) = point1
    #(x_2, y_2) = point2
    #undistort points
    left = np.array([[point1[0],point1[1]]]).reshape((1,-1,2))
    right = np.array([[point2[0],point2[1]]]).reshape((1,-1,2))
    undistorted_left = cv2.undistortPoints(left,CameraMatrix1,dist1,P=CameraMatrix1)
    undistorted_right = cv2.undistortPoints(right,CameraMatrix2,dist2,P=CameraMatrix2)
    undistorted_left = undistorted_left.reshape((-1, 2))
    undistorted_right = undistorted_right.reshape((-1, 2))
    [x_1,y_1] = undistorted_left[0]
    [x_2,y_2] = undistorted_right[0]
    disparity = x_1 - x_2
    X = round((baseline / disparity) * (x_1 - o_x - (f_skew / f_y) * (y_1 - o_y)), 2)
    Y = round(baseline * (f_x / f_y) * (y_1 - o_y) / disparity, 2)
    Z = round((f_x * baseline) / disparity, 2)
    return [X, Y, Z]