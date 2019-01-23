import cv2
import numpy as np
import time
from networktables import NetworkTables as nt

# PARAMS
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 300
params.maxArea = 6000
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
    min_i = 0
    min_j = 1
    min_score = 100000000.0
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

    return min_i, min_j


print("OpenCV version: " + cv2.__version__)

cap = cv2.VideoCapture(0)

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    ### MAIN LOOP

    r, bgr = cap.read()
    
    if r:
        start = time.time()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  
        mask = cv2.inRange(hsv, LBOUND, UBOUND)
        blurred = cv2.blur(mask, (5, 5))
        
        points = detector.detect(blurred)

        end = time.time()

        ### END MAIN LOOP

        # print("FPS: " + str(1.0 / (end - start)))

        if len(points) > 1:
            min_i, min_j = find_tapes(points)
            display = cv2.drawKeypoints(bgr, [ points[min_i], points[min_j] ], np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            sd.putNumber("tape", 0.5 * (points[min_i].pt[0] + points[min_j].pt[0]))
        
            cv2.imshow("yay", display)
    

        cv2.imshow("masked", blurred)
   
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
    print(str(LBOUND) +' '+ str(UBOUND))

cap.release()
cv2.destroyAllWindows()
