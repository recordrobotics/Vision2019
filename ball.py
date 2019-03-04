import cv2
import numpy as np
import time
from networktables import NetworkTables as nt


frameScalingFactor = 0.3

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

print("OpenCV version: " + cv2.__version__)

def kernel(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  
    mask = cv2.inRange(hsv, LBOUND, UBOUND)
    blurred = cv2.blur(mask, (5, 5))
    cv2.imshow("masked", blurred)
    
    points = detector.detect(blurred)

    return points

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440 * frameScalingFactor)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960 * frameScalingFactor)

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    ### MAIN LOOP

    r, bgr = cap.read()
    
    if r:
        start = time.time()

        points = kernel(bgr)

        end = time.time()

        ### END MAIN LOOP

#        print("FPS: " + str(1.0 / (end - start)))
	
        x = -1.5
        y = -1.5
        if len(points) > 1:
            display = cv2.drawKeypoints(bgr, points, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("yay", display)
            x = 0.0
            y = 0.0
            for p in points:
                x += 0.5 * (p.pt[0] + p.pt[0])
                y += 0.5 * (p.pt[1] + p.pt[1])
            x /= len(points)
            y /= len(points)

        print("x: " + str(x) + "  y: " + str(y))
        sd.putNumber("ball_x|PI_2", x)
        sd.putNumber("ball_y|PI_2", y)

   
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
