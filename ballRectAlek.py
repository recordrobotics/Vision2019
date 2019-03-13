
import cv2
import numpy as np
import time
from networktables import NetworkTables as nt

frameScalingFactor = 0.5

# PARAMS
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 10000#*frameScalingFactor*frameScalingFactor
params.maxArea = 1000000#*frameScalingFactor*frameScalingFactor
params.filterByCircularity = False
params.filterByColor = False
params.blobColor = 255
params.filterByConvexity = False
params.minConvexity = 0.4
params.maxConvexity = 1.0
params.filterByInertia = False
params.minInertiaRatio = 0.0
params.maxInertiaRatio = 0.4
detector = cv2.SimpleBlobDetector_create(params)

LBOUND_ORANGE = np.array([0, 100, 200])
UBOUND_ORANGE = np.array([50, 255, 255])

LBOUND_BRIGHT = np.array([0, 0, 250])
UBOUND_BRIGHT = np.array([60, 100, 255])

LBOUND_WHITE_BGR = np.array([254, 254, 254])
UBOUND_WHITE_BGR = np.array([255, 255, 255])

# END PARAMS

# ALEK STUFF
ballGamma = 0.9
ballAvgX = 0
ballAvgY = 0

print("OpenCV version: " + cv2.__version__)

def kernel(bgr):
    bgr = cv2.blur(bgr, (11, 11))
    mask_white = cv2.inRange(bgr, LBOUND_WHITE_BGR, UBOUND_WHITE_BGR)
    
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_bright = cv2.inRange(hsv, LBOUND_BRIGHT, UBOUND_BRIGHT)
    mask_orange = cv2.inRange(hsv, LBOUND_ORANGE, UBOUND_ORANGE)
    mask = cv2.bitwise_or(mask_orange, mask_bright)
    mask = cv2.bitwise_and(cv2.bitwise_not(mask_white), mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (15, 15), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (15, 15), iterations=2)
    
    cv2.imshow("Ball mask", mask)

    return mask


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap.get(cv2.CAP_PROP_FRAME_WIDTH) * frameScalingFactor)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  * frameScalingFactor)

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    ### MAIN LOOP
    r, bgr = cap.read()

    if r:
        start = time.time()
        h = bgr.shape[0]
        hw = bgr.shape[1] * 0.5
        
        mask = kernel(bgr)

        ### END MAIN LOOP
        
        ball_x = -1.5
        ball_y =  -1.5
        display_ball = bgr

        M = cv2.moments(mask)
        if abs(M["m00"]) > 0.000001:
            ball_x = (M["m10"] / M["m00"] - hw) / hw
            ball_y = (M["m01"] / M["m00"]) / h
            
            ballAvgX = ballGamma*ballAvgX + (1-ballGamma)*ball_x
            ballAvgY = ballGamma*ballAvgY + (1-ballGamma)*ball_y

            #display_ball = cv2.circle(display_ball, (int((ballAvgX + 1.0) * hw), int(ballAvgY * h)), 15, (0,250,0), 2)
            #display_ball = cv2.circle(display_ball, (int((ball_x + 1.0) * hw), int(ball_y * h)), 15, (0, 0,250), 2)
        
        #cv2.imshow("Ball view", display_ball)


        print("x: " + str(ball_x) + "  y: " + str(ball_y))
        sd.putNumber("ball_x|PI_2", ball_x)
        sd.putNumber("ball_y|PI_2", ball_y)
        
        end = time.time()
        print("Time: " + str(end - start))

    bound1 = LBOUND_BRIGHT
    bound2 = UBOUND_BRIGHT

    c = cv2.waitKey(1) & 0xFF

    keys1 = [ord(ki) for ki in "rftgyh"]
    keys2 = [ord(ki) for ki in "ujikol"]
    for k1 in range(len(keys1)):
        if c == keys1[k1]:
            dirrection = 1 if (k1 % 2)==0 else -1
            which = k1 // 2
            bound1[which] += dirrection
    for k2 in range(len(keys2)):
        if c == keys2[k2]:
            dirrection = 1 if (k2 % 2)==0 else -1
            which = k2 // 2
            bound2[which] += dirrection

    if c == ord('q'):
        break

    print(str(bound1) +' '+ str(bound2))

cap.release()
cv2.destroyAllWindows()
