import cv2
import numpy as np
import time
from networktables import NetworkTables as nt


frameScalingFactor = 0.3

# PARAMS
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 60#*frameScalingFactor*frameScalingFactor
params.maxArea = 60000#*frameScalingFactor*frameScalingFactor
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

LBOUND = np.array([50, 0, 250])
UBOUND = np.array([75, 10, 255])
# END PARAMS

# THIS IS ALEKS NEW FIND TAPE THING
# IT FINDS MULTIPLE TAPES
# THERE IS A FAIRLY GOOD CHANCE THAT MANY ARE NOT LEGIT i.e. (100,-1,-1)
HIGHEST_ACCEPTABLE_SCORE = 1.0
def find_tapes(points):
    H = 1 # 15 is 6 choose 2, it is the maxmimum number of pair that we are willing to output
    # goodPoints[i] = (pair's score, pair's left most index, pair's right most index)
    goodPoints = [(HIGHEST_ACCEPTABLE_SCORE,-1,-1) for i in range(15)]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            s = points[i].size + points[j].size
            score = abs(points[i].pt[1] - points[j].pt[1]) / s + abs(points[i].size - points[j].size) / s

            m = 0
            #print(score)
            if points[i].pt[1] > m and points[j].pt[1] > m and score < goodPoints[-1][0]:
                for k in range(H):  # NOTE: ALEK: change this to be like insertion sort so that it can be cythonated, atm it is kinda trash and sketch and only would work in python
                    if score < goodPoints[k][0]:
                        # flip if i is not the left most one
                        if points[i].pt[0] > points[j].pt[0]:
                            goodPoints.insert(k, (score, j, i))
                        else:
                            goodPoints.insert(k, (score, i, j))
                        goodPoints.pop()
                        break
    for i in range(H):
        if goodPoints[-1][2] < 0: # not real
            goodPoints.pop()
        else:
            break

    return goodPoints


print("OpenCV version: " + cv2.__version__)

def kernel(bgr, lbound, ubound, detect, blur_size = (5, 5)):
    blurred = cv2.blur(bgr, blur_size)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  
    mask = cv2.inRange(hsv, lbound, ubound)
    
    cv2.imshow("masked", mask)
    
    points = detect.detect(mask)

    return points, mask

cap = cv2.VideoCapture(1) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440 * frameScalingFactor)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960 * frameScalingFactor)

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    ### MAIN LOOP

    r, bgr = cap.read()
    
    if r:
        start = time.time()

        points, mask = kernel(bgr, LBOUND, UBOUND, detector, (3, 3))

        end = time.time()

        ### END MAIN LOOP

#        print("FPS: " + str(1.0 / (end - start)))
	
        x = -1.0
        if len(points) > 1:
            goodPoints = find_tapes(points)
            for t in goodPoints:
                min_i = t[1]
                min_j = t[2]
                
                f = 0.7
                topleft1 = (int(points[min_i].pt[1] - points[min_i].size * f), int(points[min_i].pt[0] - points[min_i].size * f))
                bottomright1 = (int(points[min_i].pt[1] + points[min_i].size * f), int(points[min_i].pt[0] + points[min_i].size * f))
                
                topleft2 = (int(points[min_j].pt[1] - points[min_j].size * f), int(points[min_j].pt[0] - points[min_j].size * f))
                bottomright2 = (int(points[min_j].pt[1] + points[min_j].size * f), int(points[min_j].pt[0] + points[min_j].size * f))

                rect1 = mask[topleft1[0]:bottomright1[0],topleft1[1]:bottomright1[1]]
                rect2 = mask[topleft2[0]:bottomright2[0],topleft2[1]:bottomright2[1]]

                indx1 = np.where(rect1 != 0)
                indx2 = np.where(rect2 != 0)

                try:
                    slope1 = np.polyfit(indx1[0], indx1[1],1)[1]
                    slope2 = np.polyfit(indx2[0], indx2[1],1)[1]
                    
                    cv2.imshow("rect1", rect1)
                    cv2.imshow("rect2", rect2)
                        
                    if slope1 < 0 and slope2 > 0: # if they point in opposite dirrections
                        display = cv2.drawKeypoints(bgr, [ points[min_i], points[min_j] ], np.array([]), (0, 0, 255),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        x = 0.5 * (points[min_i].pt[0] + points[min_j].pt[0])
                        #x = (x - 720 * frameScalingFactor) / (720 * frameScalingFactor)
                        cv2.imshow("yay", display)
                        break
                except:
                    continue
        #print(x)    

   
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
    #print(str(LBOUND) +' '+ str(UBOUND))

cap.release()
cv2.destroyAllWindows()

