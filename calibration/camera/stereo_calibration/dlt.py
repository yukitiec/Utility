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

class DLT:
    def __init__(self, cam_left_param, cam_right_param):
        self.p_left = self.get_projection_matrix(cam_left_param)

        self.p_right = self.get_projection_matrix(cam_right_param)

    def make_homogeneous_rep_matrix(self, R, t):
        """make homogeneous matrix

        Args:
            R (_type_): rotation matrix
            t (_type_): translation matrix

        Returns:
            _type_: homogeneous matrix
        """
        P = np.zeros((4, 4))
        P[:3, :3] = R
        P[:3, 3] = t.reshape(3)
        P[3, 3] = 1
        return P

    def get_projection_matrix(self, param):
        # read camera parameters
        # cmtx, dist = read_camera_parameters(camera_id) #camera matrix, distortion matrix
        # rvec, tvec = read_rotation_translation(camera_id) #rotation matrix, translation vectors
        cmtx = param[0]
        dist = param[1]
        rvec = param[2]
        tvec = param[3]
        # calculate projection matrix
        P = cmtx @ self.make_homogeneous_rep_matrix(rvec, tvec)[:3, :]
        return P

    # direct linear transform
    def main(self, point1, point2):
        """calculate 3d positions

        Args:
            P1 (_type_): projection matrix left
            P2 (_type_): projection matrix right
            point1 (_type_): point in left
            point2 (_type_): point in right

        Returns:
            _type_: 3d points (X,Y,Z)
        """
        A = [
            point1[1] * self.p_left[2, :] - self.p_left[1, :],
            self.p_left[0, :] - point1[0] * self.p_left[2, :],
            point2[1] * self.p_right[2, :] - self.p_right[1, :],
            self.p_right[0, :] - point2[0] * self.p_right[2, :],
        ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)

        # print('Triangulated point: ')
        # print(Vh[3,0:3]/Vh[3,3])
        return Vh[3, 0:3] / Vh[3, 3]