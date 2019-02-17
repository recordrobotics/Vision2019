import cv2
import numpy as np
import pickle

print(cv2.__version__)

board_width = 6
board_height = 4

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
subpix_window_size = (11, 11)

stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
stereo_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS

l_test_points = np.array([[-10], [0]], dtype=np.float32)
r_test_points = np.array([[10], [0]], dtype=np.float32)


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

#try:
for i in range(20, 230):
    print("processing " + str(i))
    l_img = cv2.imread("./images/left/" + str(i) + "img.jpg")
    r_img = cv2.imread("./images/right/" + str(i) + "img.jpg")
    
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
        
        #cv2.imshow("left", l_img)
        #cv2.imshow("right", r_img)
        
        #cv2.waitKey(5000)
        #cv2.destroyAllWindows()
#except:
#    pass


l_ret, l_mtx, l_dist, l_rvecs, l_tvecs = cv2.calibrateCamera(object_points_list, left_image_points_list, l_gray.shape[::-1], None, None)
r_ret, r_mtx, r_dist, r_rvecs, r_tvecs = cv2.calibrateCamera(object_points_list, right_image_points_list, r_gray.shape[::-1], None, None)

#print(object_points_list)
#print(left_image_points_list)
#print(right_image_points_list.total())

T = np.zeros((3, 1), dtype=np.float64)
R = np.eye(3, dtype=np.float64)
ret, l_mtx2, l_dist2, r_mtx2, r_dist2, r, t, e, f = cv2.stereoCalibrate(object_points_list, left_image_points_list, right_image_points_list, \
            l_mtx, l_dist, r_mtx, r_dist, l_gray.shape[::-1], None, None, criteria=stereo_criteria, flags=stereo_flags)

l_rmtx, r_rmtx, l_pmtx, r_pmtx, q, roi1, roi2 = cv2.stereoRectify(l_mtx, l_dist, r_mtx, r_dist, l_gray.shape[::-1], r, t, alpha=0.0)

print(l_pmtx)

projected_points = cv2.triangulatePoints(l_pmtx, r_pmtx, l_test_points, r_test_points)

pickle.dump({ "l": l_pmtx, "r": r_pmtx }, open("pmtxs.data", "wb"))

print(projected_points)
