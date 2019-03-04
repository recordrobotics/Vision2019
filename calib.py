import cv2
import numpy as np
import pickle

print(cv2.__version__)

board_width = 8
board_height = 4

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
subpix_window_size = (11, 11)

stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
stereo_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS
rectify_alpha = 0.0


left_cap = cv2.VideoCapture(1)
right_cap = cv2.VideoCapture(2)

object_points = np.zeros((board_width * board_height, 3), np.float32)
object_points[:,:2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

#object_points = np.zeros( (board_width * board_height, 3), np.float32 )
#object_points[:,:2] = np.indices([ board_width, board_height]).T.reshape(-1, 2)

print(object_points)

object_points_list = []

left_image_points_list = []
right_image_points_list = []

img = None
l_gray = None
r_gray = None

i = 1
try:
#for i in [48, 49, 58, 129]:
    while 1:
        left_cap.grab()
        right_cap.grab()

        l_r, l_img = left_cap.retrieve()
        r_r, r_img = right_cap.retrieve()
        
        if l_r and r_r:
            print("processing " + str(i))
            i += 1
            #l_img = cv2.imread("./images/left/" + str(i) + "img.jpg")
            #r_img = cv2.imread("./images/right/" + str(i) + "img.jpg")
            
            l_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
            r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

            l_s2, l_corners = cv2.findChessboardCorners(l_gray, (board_width, board_height), None)
            r_s2, r_corners = cv2.findChessboardCorners(r_gray, (board_width, board_height), None)

            if l_s2 and r_s2:
                object_points_list.append(object_points)

                l_corners2 = cv2.cornerSubPix(l_gray, l_corners, subpix_window_size, (-1, -1), subpix_criteria)
                r_corners2 = cv2.cornerSubPix(r_gray, r_corners, subpix_window_size, (-1, -1), subpix_criteria)
                
                left_image_points_list.append(l_corners2)
                right_image_points_list.append(r_corners2)

                print("new images")

                cv2.drawChessboardCorners(l_img, (board_width, board_height), l_corners2, True)
                cv2.drawChessboardCorners(r_img, (board_width, board_height), r_corners2, True)
                
            cv2.imshow("left", l_img)
            cv2.imshow("right", r_img)
                
            cv2.waitKey(100)
except:
    pass

cv2.destroyAllWindows()

l_ret, l_mtx, l_dist, l_rvecs, l_tvecs = cv2.calibrateCamera(object_points_list, left_image_points_list, l_gray.shape[::-1], None, None)
r_ret, r_mtx, r_dist, r_rvecs, r_tvecs = cv2.calibrateCamera(object_points_list, right_image_points_list, r_gray.shape[::-1], None, None)

#print(object_points_list)
#print(left_image_points_list)
#print(right_image_points_list.total())

ret, l_mtx, l_dist, r_mtx, r_dist, r, t, e, f = cv2.stereoCalibrate(object_points_list, left_image_points_list, right_image_points_list, \
            l_mtx, l_dist, r_mtx, r_dist, l_gray.shape[::-1], None, None, criteria=stereo_criteria, flags=stereo_flags)

l_rmtx, r_rmtx, l_pmtx, r_pmtx, q, roi1, roi2 = cv2.stereoRectify(l_mtx, l_dist, r_mtx, r_dist, l_gray.shape[::-1], r, t, alpha=rectify_alpha)

print(l_pmtx)

pickle.dump({ "r": r, "t": t, "q": q, "lr": l_rmtx, "rr": r_rmtx, "lp": l_pmtx, "rp": r_pmtx, "rm": r_mtx, "lm": l_mtx, "ld": l_dist, "rd": r_dist, "l_roi": roi1, "r_roi": roi2 }, 
                open("pmtxs.data", "wb"))

print(roi1)
print(roi2)
