import cv2
import numpy as np
import yaml
import time
import cv2
from networktables import NetworkTables as nt


frameScalingFactor = 1.0

# PARAMS
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 60#*frameScalingFactor*frameScalingFactor
params.maxArea = 6000#*frameScalingFactor*frameScalingFactor
params.filterByCircularity = False
params.filterByColor = False
params.blobColor = 255
params.filterByConvexity = True
params.minConvexity = 0.3
params.maxConvexity = 1.0
params.filterByInertia = True
params.minInertiaRatio = 0.04
params.maxInertiaRatio = 0.5
detector = cv2.SimpleBlobDetector_create(params)

LBOUND = np.array([45, 0, 250])
UBOUND = np.array([75, 100, 255])
# END PARAMS

def find_tapes(points):
    min_i = -1
    min_j = -1
    min_score = 0.1
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            s = points[i].size + points[j].size
            score = abs(points[i].pt[1] - points[j].pt[1]) / s + \
                    abs(points[i].size - points[j].size) / s

            m = 0
            if points[i].pt[1] > m and points[j].pt[1] > m and score < min_score:
                min_i = i
                min_j = j
                min_score = score

#    print("Score: " + str(min_score))
    return min_i, min_j


print("OpenCV version: " + cv2.__version__)


l_cap = cv2.VideoCapture(2)
#l_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440 * frameScalingFactor)
#l_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960 * frameScalingFactor)

r_cap = cv2.VideoCapture(1)
#r_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440 * frameScalingFactor)
#r_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960 * frameScalingFactor)

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

d2 = yaml.load(open("stereo_calib", "r"))

l_pmtx = np.array(d2["P1"]["data"]).reshape(3, 4) #d["lp"]
r_pmtx = np.array(d2["P2"]["data"]).reshape(3, 4) #d["rp"]
l_rmtx = np.array(d2["R1"]["data"]).reshape(3, 3) #d["lr"]
r_rmtx = np.array(d2["R2"]["data"]).reshape(3, 3) #d["lr"]
r_mtx = np.array(d2["K2"]["data"]).reshape(3, 3) #d["lr"]
l_mtx = np.array(d2["K1"]["data"]).reshape(3, 3) #d["lr"]
r_dist = np.array(d2["D2"]["data"]) #d["lr"]
l_dist = np.array(d2["D1"]["data"]) #d["lr"]
t = np.array(d2["T"]) #d["lr"]
r = np.array(d2["R"]["data"]).reshape(3, 3) #d["lr"]

while 1:
    ### MAIN LOOP

    l_cap.grab()
    r_cap.grab()

    r, l_bgr = l_cap.retrieve()
    r2, r_bgr = r_cap.retrieve()
    
    if r and r2:
        l_hsv = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2HSV)  
        l_mask = cv2.inRange(l_hsv, LBOUND, UBOUND)
        l_blurred = cv2.blur(l_mask, (5, 5))
        
        l_points = detector.detect(l_blurred)

        ### END MAIN LOOP

#        print("FPS: " + str(1.0 / (end - start)))
	
        l_x = -1.0
        l_y = -1.0
        if len(l_points) > 1:
            min_i, min_j = find_tapes(l_points)
            if min_i >= 0:
#                display = cv2.drawKeypoints(bgr, [ points[min_i], points[min_j] ], np.array([]), (0, 0, 255),
#                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                l_x = 0.5 * (l_points[min_i].pt[0] + l_points[min_j].pt[0])
                l_y = 0.5 * (l_points[min_i].pt[1] + l_points[min_j].pt[1])
                #x = (x - 720 * frameScalingFactor) / (720 * frameScalingFactor)
#                cv2.imshow("yay", display)
        
        r_hsv = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2HSV)  
        r_mask = cv2.inRange(r_hsv, LBOUND, UBOUND)
        r_blurred = cv2.blur(r_mask, (5, 5))
        
        r_points = detector.detect(r_blurred)

        ### END MAIN LOOP

#        print("FPS: " + str(1.0 / (end - start)))
	
        r_x = -1.0
        r_y = -1.0
        if len(r_points) > 1:
            min_i, min_j = find_tapes(r_points)
            if min_i >= 0:
#                display = cv2.drawKeypoints(bgr, [ points[min_i], points[min_j] ], np.array([]), (0, 0, 255),
#                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                r_x = 0.5 * (r_points[min_i].pt[0] + r_points[min_j].pt[0])
                r_y = 0.5 * (r_points[min_i].pt[1] + r_points[min_j].pt[1])
                #x = (x - 720 * frameScalingFactor) / (720 * frameScalingFactor)
#                cv2.imshow("yay", display)

        if r_x != -1.0 and l_x != -1.0:
            print("points: " + str(l_x) + " " + str(r_x))    
            
            l_test_points = np.array([[[l_x, l_y]]])
            r_test_points = np.array([[[r_x, r_y]]])

            l_undistorted = cv2.undistortPoints(l_test_points, l_mtx, l_dist, None, l_rmtx, l_pmtx)
            r_undistorted = cv2.undistortPoints(r_test_points, r_mtx, r_dist, None, r_rmtx, r_pmtx)

            print("\nleft undistorted: " + str(l_undistorted))
            print("\nright undistorted: " + str(r_undistorted))

            #projected_points = cv2.triangulatePoints(l_pmtx, r_pmtx, l_test_points, r_test_points)
            projected_points2 = cv2.triangulatePoints(l_pmtx, r_pmtx, l_undistorted, r_undistorted)

            #print("\nElapsed: " + str(end - start))

            #print("\nProjected: " + str((projected_points[:3] / projected_points[3]).T))
            print("\nProjected2: " + str((projected_points2[:3] / projected_points2[3]).T))


#        cv2.imshow("masked", blurred)
   
    c = cv2.waitKey(1) & 0xFF
    if c == ord('r'):
        LBOUND[0] += 1
    elif c == ord('f'):
        LBOUND[0] -= 1
    elif c == ord('t'):
        LBOUND[1] += 1
    elif c == ord('g'):
        LBOUND[1] -= 1
    elif c == ord('y'):
        LBOUND[2] += 1
    elif c == ord('h'):
        LBOUND[2] -= 1
    if c == ord('u'):
        UBOUND[0] += 1
    elif c == ord('j'):
        UBOUND[0] -= 1
    elif c == ord('i'):
        UBOUND[1] += 1
    elif c == ord('k'):
        UBOUND[1] -= 1
    elif c == ord('o'):
        UBOUND[2] += 1
    elif c == ord('l'):
        UBOUND[2] -= 1
    elif c == ord('q'):
        break
#    print(str(LBOUND) +' '+ str(UBOUND))

cap.release()
cv2.destroyAllWindows()
