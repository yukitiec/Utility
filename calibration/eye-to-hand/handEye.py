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

def pose2matrix(pose):
    """
    @brief convert pose into homogeneous matrix
    Args:
        pose: [x,y,z,rx,ry,rz], [rx,ry,rz] is rotation vector
    Return:
        rotation_matrix(3,3) 
        pose [x,y,z]
    """
    H_c = np.zeros((4,4))
    # Create rotation matrices for each axis// Convert angles to radians
    x = pose[0]
    y = pose[1]
    z = pose[2]
    nx = pose[3]
    ny = pose[4]
    nz = pose[5]
    #rotation angle and rotational axis
    angle = (nx^2+ny^2+nz^2)^(0.5)
    nx /= angle
    ny /= angle
    nz /= angle
	#rodrigues formula
	axisCross = np.array([[0,-nz,ny],[nz,0,-nx],[-ny,nx,0]])
    rotation_matrix = np.eye(3)+np.sin(angle)*axisCross+(1-np.cos(angle))*np.dot(axisCross,axisCross)

    #print("Rotation Matrix:")
    #print(rotation_matrix)

    H_c[:3,:3]=rotation_matrix
    H_c[:3,3] = np.array(pose[:3])
    H_c[3,3]=1
    #print("transformation matrxi=",H_c)
    return rotation_matrix, np.array(pose[:3])

def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):

    if eye_to_hand:
        # change coordinates from gripper2base to base2gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        
        
        # change parameters values
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    # calibrate
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )
    print("rotation matrix = ",R,"\ntranslation =",t)
    return R, t