import cv2
import numpy as np
import time
from networktables import NetworkTables as nt


frameScalingFactor = 0.3

# PARAMS
LBOUND_ORANGE = np.array([0, 100, 200])
UBOUND_ORANGE = np.array([50, 255, 255])

LBOUND_BRIGHT = np.array([0, 0, 250])
UBOUND_BRIGHT = np.array([60, 100, 255])

LBOUND_WHITE_BGR = np.array([254, 254, 254])
UBOUND_WHITE_BGR = np.array([255, 255, 255])
# END PARAMS

print("OpenCV version: " + cv2.__version__)

def kernel(bgr):
    bgr = cv2.blur(bgr, (7, 7))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_white = cv2.inRange(bgr, LBOUND_WHITE_BGR,UBOUND_WHITE_BGR)
    mask_bright = cv2.inRange(hsv, LBOUND_BRIGHT, UBOUND_BRIGHT)
    mask_orange = cv2.inRange(hsv, LBOUND_ORANGE, UBOUND_ORANGE)
    mask = cv2.bitwise_or(mask_orange, mask_bright)
    mask = cv2.bitwise_and(cv2.bitwise_not(mask_white),mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (31, 31), iterations=2)

    return mask

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap.get(cv2.CAP_PROP_FRAME_WIDTH) * frameScalingFactor)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  * frameScalingFactor)

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    ### MAIN LOOP

    r, bgr = cap.read()
    h = bgr.shape[0]
    hw = bgr.shape[1] * 0.5
    
    if r:
        start = time.time()

        mask = kernel(bgr)

        end = time.time()

        ### END MAIN LOOP

        cv2.imshow("masked", mask)

        try:
            M = cv2.moments(mask)
            x = (M["m10"] / M["m00"] - hw) / hw
            y = (M["m01"] / M["m00"]) / h
        
            print("x: " + str(x) + "  y: " + str(y))
            sd.putNumber("ball_x|PI_2", x)
            sd.putNumber("ball_y|PI_2", y)
        except:
            sd.putNumber("ball_x|PI_2", -2)
            sd.putNumber("ball_y|PI_2", -2)


   
    bound1 = LBOUND_ORANGE
    bound2 = UBOUND_ORANGE

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
