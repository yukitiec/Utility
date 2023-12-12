"""stereo.py
Content:
 class StereoVision:
    calibration() : calibration of 2 cameras -> get inner matrix
    qualityCalibration() : check quality of calibration
    undistort() : undistort images
    reconstruction() : 2d points -> 3d points    
"""
import numpy as np
import cv2
import os
import statistics
import pandas as pd
import random


class StereoVision:
    def __init__(
        self, baseline, checker_board, check_size, start, end, criteria, src, savePath
    ):
        self.baseline = baseline  # distance between 2 cameras
        self.checker_board = checker_board
        self.check_size = check_size
        self.criteria = criteria
        self.start = start
        self.end = end
        # calibration images path
        self.src = src

        # calibration result [ret1, mtx1, dist1, rvecs1, tvecs1, objp_right,imgpoints_right]
        # print("--- Left ---\n")
        # self.calibrate_left = self.calibration(cam="left")
        # print("--- Right ---\n")
        # self.calibrate_right = self.calibration(cam="right")
        self.result_calibration = self.stereoCalibrate()
        self.K_left = self.result_calibration[0]
        self.dist_left = self.result_calibration[1]
        self.K_right = self.result_calibration[2]
        self.dist_right = self.result_calibration[3]
        self.rotation = self.result_calibration[4]
        self.translation = self.result_calibration[5]
        self.rotation_left = self.result_calibration[6]
        self.rotation_right = self.result_calibration[7]
        self.projection_left = self.result_calibration[8]
        self.projection_right = self.result_calibration[9]
        self.Q = self.result_calibration[10]

        # extrinsic matrix

        # focal lenght in pixel
        self.f_x = (self.K_left[0][0] + self.K_right[0][0]) / 2
        self.f_y = (self.K_left[1][1] + self.K_right[1][1]) / 2
        self.f_skew = (self.K_left[0][1] + self.K_right[0][1]) / 2
        self.o_x = (self.K_left[0][2] + self.K_right[0][2]) / 2
        self.o_y = (self.K_left[1][2] + self.K_right[1][2]) / 2

        # write data to csv file
        data = {
            "camera": ["Left", "Right"],
            "camera Matrix": [self.K_left, self.K_right],
            "distortion coeffs": [self.dist_left, self.dist_right],
            "rotation": [np.eye(3, dtype=np.float32), self.rotation],
            "translation": [
                np.array([0.0, 0.0, 0.0]).reshape((3, 1)),
                self.translation,
            ],
            "Rotation matrix": [self.rotation_left, self.rotation_right],
            "projection matrix": [self.projection_left, self.projection_right],
            "Q matrix": [self.Q, self.Q],
        }
        df = pd.DataFrame(data)
        file_name = savePath
        df.to_csv(savePath, index=False)

    def calibration(self, cam):
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros(
            (1, self.checker_board[0] * self.checker_board[1], 3), np.float32
        )
        # objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0]*CHECK_SIZE:CHECK_SIZE, 0:CHECKERBOARD[1]*CHECK_SIZE:CHECK_SIZE].T.reshape(-1, 2) #consider checker board size
        objp[0, :, :2] = np.mgrid[
            0 : self.checker_board[0], 0 : self.checker_board[1]
        ].T.reshape(-1, 2)
        objp = self.check_size * objp
        # print(objp)
        prev_img_shape = None

        # Extracting path of individual image stored in a given director
        for i in range(self.start, self.end):
            filename = "{cam}/{idx}.png".format(cam=cam, idx=i)
            if i % 100 == 0:
                print(filename)
            img = cv2.imread(os.path.join(self.src, filename))
            if len(img.shape) == 3:
                # print(img.shape)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.checker_board,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of checker board
        """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(
                    gray, corners, (9, 9), (-1, -1), self.criteria
                )

                imgpoints.append(corners2)

                # if i % 50 == 0:
                # print("{}".format(i))
                # Draw and display the corners
                # img = cv2.drawChessboardCorners(img, self.checker_board, corners2, ret)

                # show detected results
                # cv2.imshow(f"img{i}", img)

        h, w = img.shape[:2]

        """
        Performing camera calibration by
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the
        detected corners (imgpoints)

        mtx:cameraMatrix(Intrinsic camera matrix)
        dist:distCoeffs:Lens coefficients
        rvecs:Rotation specified as a 3*1 vector, vector : the axis of rotation, magnitude : angle of rotarion
        tvecs : translation vector
        """

        # gray.shape[::-1]:inversed order (y,x) -> (x,y)
        # rvecs : rotation vectors tvecs : translation vectors
        # intrinsic camera matrix
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        print("RMSE : \n", ret)
        print("Camera matrix : \n")
        print(mtx)
        print("distCoeffs : \n")
        print(dist)
        return [ret, mtx, dist, rvecs, tvecs, objp, imgpoints]

    def qualityCalibration(self, calibrate_data):
        """Check Calibration Quality"""
        # import calibration data
        [ret, mtx, dist, rvecs, tvecs, objp, imgpoints] = calibrate_data
        mean_errors = []
        for j in range(
            0, len(imgpoints)
        ):  # for every images : len(imgpoints_left) = num of images
            tot_error = 0
            for i in range(len(objp[0])):  # for every points
                [imgpoints2], _ = cv2.projectPoints(
                    objp[0][i], rvecs[j], tvecs[j], mtx, dist
                )
                error = cv2.norm(imgpoints[j][i], imgpoints2, cv2.NORM_L2) / len(
                    imgpoints2
                )  # (err_x^2*err_y^2)/2
                tot_error += error
            mean_error = tot_error / len(objp[0])  # error for one points
            mean_errors.append(mean_error)
            print(
                "{} :: average error for one edge : {:.2f} [pixel]".format(
                    j + 1, mean_error
                )
            )

        ave_error = statistics.mean(mean_errors)
        print("average errors : {:.2f} [pixel]".format(ave_error))

    def stereoCalibrate(self):
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints_left = []
        imgpoints_right = []

        # Defining the world coordinates for 3D points
        objp = np.zeros(
            (1, self.checker_board[0] * self.checker_board[1], 3), np.float32
        )
        # objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0]*CHECK_SIZE:CHECK_SIZE, 0:CHECKERBOARD[1]*CHECK_SIZE:CHECK_SIZE].T.reshape(-1, 2) #consider checker board size
        objp[0, :, :2] = np.mgrid[
            0 : self.checker_board[0], 0 : self.checker_board[1]
        ].T.reshape(-1, 2)
        objp = self.check_size * objp
        # print(objp)
        prev_img_shape = None

        # Extracting path of individual image stored in a given director
        for i in range(self.start, self.end):
            filename_left = "left/{idx}.png".format(idx=i)
            filename_right = "right/{idx}.png".format(idx=i)
            if i % 100 == 0:
                print(filename_left, filename_right)
            img_left = cv2.imread(os.path.join(self.src, filename_left))
            img_right = cv2.imread(os.path.join(self.src, filename_right))
            if random.random() > 0.6:
                if len(img_left.shape) == 3:
                    # print(img.shape)
                    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                else:
                    gray_left = img_left
                if len(img_right.shape) == 3:
                    # print(img.shape)
                    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
                else:
                    gray_right = img_right
                # Find the chess board corners
                # If desired number of corners are found in the image then ret = true
                ret_left, corners_left = cv2.findChessboardCorners(
                    gray_left,
                    self.checker_board,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK
                    + cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                ret_right, corners_right = cv2.findChessboardCorners(
                    gray_right,
                    self.checker_board,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK
                    + cv2.CALIB_CB_NORMALIZE_IMAGE,
                )

                """
                If desired number of corner are detected,
                
                we refine the pixel coordinates and display
                them on the images of checker board
                """
                if ret_left == True and ret_right == True:
                    objpoints.append(objp)
                    # refining pixel coordinates for given 2d points.
                    corners2_left = cv2.cornerSubPix(
                        gray_left, corners_left, (7, 7), (-1, -1), self.criteria
                    )
                    corners2_right = cv2.cornerSubPix(
                        gray_right, corners_right, (7, 7), (-1, -1), self.criteria
                    )
                    imgpoints_left.append(corners2_left)
                    imgpoints_right.append(corners2_right)

                    # if i % 50 == 0:
                    # print("{}".format(i))
                    # Draw and display the corners
                    # img = cv2.drawChessboardCorners(img, self.checker_board, corners2, ret)

                    # show detected results
                    # cv2.imshow(f"img{i}", img)

        h, w = img_left.shape[:2]
        # intrinsic matrix
        ret, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, gray_left.shape[::-1], None, None
        )
        print("RMSE : \n", ret)
        print("Camera matrix : \n", mtx_left)
        print("distCodffs : \n", dist_left)
        ret, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, gray_left.shape[::-1], None, None
        )
        print("RMSE : \n", ret)
        print("Camera matrix : \n", mtx_right)
        print("distCodffs : \n", dist_right)
        # show result
        print("reprojection data")
        print("!!LEFT!!")
        self.qualityCalibration(
            calibrate_data=[
                ret,
                mtx_left,
                dist_left,
                rvecs_left,
                tvecs_left,
                objp,
                imgpoints_left,
            ]
        )
        print("!!RIGHT!!")
        self.qualityCalibration(
            calibrate_data=[
                ret,
                mtx_right,
                dist_right,
                rvecs_right,
                tvecs_right,
                objp,
                imgpoints_right,
            ]
        )
        # extrinsic matrix
        print(gray_left.shape[::-1])
        (
            ret,
            mtx2_left,
            dist2_left,
            mtx2_right,
            dist2_right,
            R,
            T,
            E,
            F,
        ) = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            mtx_left,
            dist_left,
            mtx_right,
            dist_right,
            gray_left.shape[::-1],
            criteria=self.criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        print("Rotation matrix : ", R)
        print("translation matrix :", T)
        print("reprojection data 2")
        print("!!LEFT!!")
        self.qualityCalibration(
            calibrate_data=[
                ret,
                mtx2_left,
                dist2_left,
                rvecs_left,
                tvecs_left,
                objp,
                imgpoints_left,
            ]
        )
        print("!!RIGHT!!")
        self.qualityCalibration(
            calibrate_data=[
                ret,
                mtx2_right,
                dist2_right,
                rvecs_right,
                tvecs_right,
                objp,
                imgpoints_right,
            ]
        )
        # calculate rotation matrix, projection matrix and Q
        (
            R_left,
            R_right,
            P_left,
            P_right,
            Q,
            validPixROI1,
            validPixRPI2,
        ) = cv2.stereoRectify(
            mtx2_left, dist2_left, mtx2_right, dist2_right, gray_left.shape[::-1], R, T
        )
        print("Rotation matrix :: LEFT=", R_left)
        print("Rotation matrix :: RIGHT=", R_right)
        print("Projection matrix :: LEFT=", P_left)
        print("Projection matrix :: RIGHT=", P_right)
        print("Q=", Q)
        # R_left, R_right : rotation matrix
        # P_left, P_right : projection matrix in the new rectified coordinate systemys
        # Q : 4*4 disparity-to-depth mapping matrix
        return [
            mtx2_left,
            dist2_left,
            mtx2_right,
            dist2_right,
            R,
            T,
            R_left,
            R_right,
            P_left,
            P_right,
            Q,
        ]
