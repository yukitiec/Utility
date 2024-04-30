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

class PoseEstimation():
    def __init__(self,checkerBoard,checkersize):
        print("construct Chessboard pose estimation class")
        self.checkerBoard=checkerBoard
        self.checkersize=checkersize

    #get x,y,z axis
    def draw(self,img, corners, imgpts,boolReverse=False):
        corner = tuple(int(i) for i in corners[0].ravel())
        print(imgpts)
        #print("axis points:")
        img = cv2.line(img, corner, tuple(int(i) for i in imgpts[0].ravel()), (255,0,0), 2) #y
        cv2.circle(img,tuple(int(i) for i in imgpts[0].ravel()),radius=7,color=(255,0,0),thickness=1)
        img = cv2.line(img, corner, tuple(int(i) for i in imgpts[1].ravel()), (0,255,0), 2) #x
        cv2.circle(img,tuple(int(i) for i in imgpts[1].ravel()),radius=7,color=(0,255,0),thickness=1)
        img = cv2.line(img, corner, tuple(int(i) for i in imgpts[2].ravel()), (0,0,255), 2) #z
        cv2.circle(img,tuple(int(i) for i in imgpts[2].ravel()),radius=7,color=(0,0,255),thickness=1)
        return img

    def vec2Matrix(self,rvec,tvec):
        """from translational and rotational vector to homogeneous matrix

        Args:
            rvec (_type_): rotational vector
            tvec (_type_): translational vector
        Return:
            Homogeneous matrix
        """
        (R, jac) = cv2.Rodrigues(rvec)
        M = np.eye(4)
        M[0:3, 0:3] = R
        M[0:3,3] = tvec.reshape(3)
        return M

    def getPose(self,cameraMatrix,dist,src,rootDir,center_based=False,debug=True,boolReverse=False):
        """get axis-end points from chessboard 
        Args:
            cameraMatrix (np.array): camera matrix
            dist (np.array): distortion coefficient
            src (char): image path
            checkerBoard (tuple): checker board size Defaults to CHECKERBOARD.
            center_based (bool) : if coordinates origin is the center of the image
        Returns:
            rotation and translation 
        """
        # setting 
        #print(src)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        objp = np.zeros((self.checkerBoard[0]*self.checkerBoard[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.checkerBoard[0],0:self.checkerBoard[1]].T.reshape(-1,2)
        for i in range(3):
            for j in range(4):
                objp[i*4+j][0] = i
                objp[i*4+j][1] = j
        #objp = objp.transpose(1,0)
        objp = self.checkersize * objp
        # 3d axis setting : (x,y,z)
        axis = np.float32([[self.checkersize,0,0], [0,self.checkersize,0], [0,0,self.checkersize]]).reshape(-1,3)
        #print("index:",index)
        img = cv2.imread(src)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerBoard,None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),criteria)
            print(corners.shape)
            if center_based:
                for i in range(corners.shape[0]):
                    corners2[i][0][0] -= cameraMatrix[0][2] #convert to center coordinates : o_x = cameraMatrix[0][2]
                    corners2[i][0][1] -= cameraMatrix[1][2] #o_y = cameraMatrix[1][2]
            #print(corners2)
            if boolReverse:
                corners2 = corners2[::-1]
                #for 20240405
                #corners_new = np.zeros((corners2.shape[0],1,2))
                #for i in range(3):
                #    for j in range(4):
                #        corners_new[i*4+j]=corners2[4*(i+1)-1-j]
                #corners2 = corners_new
            # Find the rotation and translation vectors.
            ret,rvec, tvec= cv2.solvePnP(objp, corners2, cameraMatrix, dist)
            transMat = self.vec2Matrix(rvec,tvec)
            print("corners2=\n",corners2,"\n")
            print("objp=\n",objp,"\n")
            if debug:
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix, dist)
                #original points of checker board
                original = tuple(int(i) for i in corners2[0].ravel())
                yEnd2 = tuple(int(i) for i in corners2[1].ravel())
                xEnd2 = tuple(int(i) for i in corners2[self.checkerBoard[0]].ravel())
                #x-axis end point
                xEnd = tuple(i for i in imgpts[1].ravel())
                yEnd = tuple(i for i in imgpts[0].ravel())
                zEnd = tuple(i for i in imgpts[2].ravel())
                cv2.circle(img,original,radius=3,color=(255,255,255),thickness=-1)
                cv2.circle(img,yEnd2,radius=3,color=(255,255,255),thickness=-1)
                cv2.circle(img,xEnd2,radius=3,color=(255,255,255),thickness=-1)
                #print(f"original points : {original}, x-end : {xEnd}, y-end : {yEnd}, z-end : {zEnd}")
                #imgpts = imgpts.squeeze()
                #print(imgpts)
                #print(tuple(int(i) for i in imgpts[0].ravel()))
                #draw 3d matrix
                img = self.draw(img,corners2,imgpts,boolReverse=boolReverse)
                #img = draw2(img,corners2,xEnd2,yEnd2,boolReverse=boolReverse)
                #save image
                index = int((src.split("\\")[-1]).split(".")[0])
                saveDir = os.path.join(rootDir,"corners")
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                cv2.imwrite(os.path.join(saveDir,f"{index:03d}.png"),img)
            return transMat
        else:
            print("can't find all corners")
            return np.eye(4)


    def undistortImg(self,src,K,dist):
        """undistort image

        Args:
            src (_type_): _description_
            K (_type_): camera matrix
            dist (_type_): distortion coeffs
            saveDir (_type_): _description_
            pos (_type_): _description_
        Return 
            save image directory
        """
        #save rectification map?
        save = True
        #prepare save directory
        img_save_dir = os.path.join(src,"undistort")
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        #count number of file
        imgsPath = [os.path.join(src,f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        numFile = len(imgsPath)
        print(numFile)
        for file in imgsPath:
            #print(file)
            img = cv2.imread(file)
            if (len(img.shape) == 3): #RGB
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            idx = int(file.split("\\")[-1].split(".")[0])
            #print(idx)
            #print(img.shape)
            #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            gray_undistort =cv2.undistort(img, K, dist)
            #gray_undistort = undistort(img,K=K,dist=dist,R=R,P=P,saveDir=saveDir,pos=pos,save=save)
            #print(i)
            #print(gray_undisto}rt.shape)
            cv2.imwrite(os.path.join(img_save_dir,f"{idx:02d}.png"),gray_undistort)
        return img_save_dir