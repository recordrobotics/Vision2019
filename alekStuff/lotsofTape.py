
# NOTE TO vassilios and alek
# you should probably look at images of the tilted tape to see if it looks good enough to distinguish between
# once engineering makes it or whatever

# NOTES NOTES:!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!check centerScore (are all the things I am using x values?!?!)!!!!!!!!!!!!!!!!

import cv2
import numpy as np
import time
from networktables import NetworkTables as nt


frameScalingFactor = 0.8

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

LBOUND = np.array([0, 0, 0]) #np.array([50, 0, 250])
UBOUND = np.array([75, 10, 255])
# END PARAMS

# THIS IS ALEKS NEW FIND TAPE THING
# IT FINDS MULTIPLE TAPES
# THERE IS A FAIRLY GOOD CHANCE THAT MANY ARE NOT LEGIT i.e. (100,-1,-1)
LOWEST_ACCEPTABLE_SCORE = 1.0
def find_tapes(points):
    H = 15 # 15 is 6 choose 2, it is the maxmimum number of pair that we are willing to output
    # goodPoints[i] = (pair's score, pair's left most index, pair's right most index)
    goodPoints = [(LOWEST_ACCEPTABLE_SCORE,-1,-1) for i in range(15)]
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

# WRONG CAMERA BAD BAD BAD BAD BAD BAD
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

        points, mask = kernel(bgr, LBOUND, UBOUND, detector, (3, 3))

        end = time.time()

        ### END MAIN LOOP
        # print("FPS: " + str(1.0 / (end - start)))

        x = -1.0
        if len(points) > 1:
            goodPoints = find_tapes(points) # NOTE: this is sorted by score and only includes points which were accepted by the score function

            if len(goodPoints) > 0:
                best_score = goodPoints[0][0]
                for i in range(1, len(goodPoints)):
                    if goodPoints[i][0] > best_score * 5:
                        goodPoints = goodPoints[:i]
                        break
                
                
                # sort by distance
                goodPoints.sort(key = lambda curTuple: abs(points[curTuple[2]].pt[0] - points[curTuple[1]].pt[0])) # tbh dont really need abs, it is certainly positive
                
                print(goodPoints)

                # will store real canidates who did well on distance and have the correct orientation
                top3 = []

                # now compute orientation on only the top 5 (/\/\/\) 6 tapes, could have 5 at most maybe less are in field of view though
                for i in range(min(len(goodPoints), 5)):

                    # compute the orientation of the strip (very expensive(?(numpy is kinda pro so maybe not...)))
                    f = 0.7

                    topleft1 = (int(points[goodPoints[i][1]].pt[1] - points[goodPoints[i][1]].size * f), int(points[goodPoints[i][1]].pt[0] - points[goodPoints[i][1]].size * f))
                    bottomright1 = (int(points[goodPoints[i][1]].pt[1] + points[goodPoints[i][1]].size * f), int(points[goodPoints[i][1]].pt[0] + points[goodPoints[i][1]].size * f))

                    topleft2 = (int(points[goodPoints[i][2]].pt[1] - points[goodPoints[i][2]].size * f), int(points[goodPoints[i][2]].pt[0] - points[goodPoints[i][2]].size * f))
                    bottomright2 = (int(points[goodPoints[i][2]].pt[1] + points[goodPoints[i][2]].size * f), int(points[goodPoints[i][2]].pt[0] + points[goodPoints[i][2]].size * f))

                    rect1 = mask[topleft1[0]:bottomright1[0],topleft1[1]:bottomright1[1]]
                    rect2 = mask[topleft2[0]:bottomright2[0],topleft2[1]:bottomright2[1]]

                    indx1 = np.where(rect1 != 0)
                    indx2 = np.where(rect2 != 0)

                    #print(indx1)
                    slope1 = np.polyfit(indx1[0], indx1[1],1)[0] # get the a in y=ax+b
                    slope2 = np.polyfit(indx2[0], indx2[1],1)[0] # first coeffiecient of the poly

                    if slope1 * slope2 < 0: # if they point in opposite dirrections
                        top3.append(goodPoints[i])
                        if len(top3) >= 3:
                            break

                    # cv2.imshow("rect1", rect1)
                    # cv2.imshow("rect2", rect2)

                    # print(points[goodPoints[i][1]].size, points[goodPoints[i][2]].size)
                    # display = cv2.drawKeypoints(bgr, [ points[goodPoints[i][0]], points[goodPoints[i][1]] ], np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # x = 0.5 * (points[goodPoints[i][1]].pt[0] + points[goodPoints[i][2]].pt[0])
                    # x = (x - 720 * frameScalingFactor) / (720 * frameScalingFactor)
                    # cv2.imshow("yay", display)
                    # print(x)
                # now we have the "top 3"
                # we will choose based on which of them is closest to the center UNLESS (see edge case later)

                centerScores = []
                for i in range(len(top3)): # note: len(top3) need not be 3, if we didn't find 3
                    # CHECK THIS: OPENCV weirdness might make this wrong... pt[0] gives an x, and I am pretty sure that so does mask.shape[1]
                    centerScores.append((abs((points[top3[i][2]].pt[0] + points[top3[i][1]].pt[0]) * 0.5 - mask.shape[1] * 0.5), top3[i]))
                centerScores.sort(key = lambda x: x[0])

                # OK, normally I will just choose whatever has the lowest score
                # but, I want to check to make sure that it is not bad (adversary: /__\,/\__,__/\)

                dists = [abs(points[centerScores[i][1][2]].pt[0]-points[centerScores[i][1][1]].pt[0]) for i in range(len(top3))]
                print(centerScores)
                if len(dists) > 1 and dists[0] > dists[1]*1.5:
                    # OK this is a bit eXTREME
                    seekPoint = centerScores[1][1]
                elif len(dists) > 2 and dists[0] > dists[2]*1.5:
                    # OK this is a bit eXTREME
                    seekPoint = centerScores[2][1]
                elif len(dists) > 0:
                    seekPoint = centerScores[0][1]
                else:
                    seekPoint = -1

                # QUESTION: are there other edge cases??????????????

                if seekPoint != -1:
                    goal_i = seekPoint[1]
                    goal_j = seekPoint[2]
                    # GO TO THIS POINT!!!!!

                    cv2.circle(bgr, (int(points[goal_i].pt[0]), int(points[goal_i].pt[1])), 10, (0,0,0), 5)
                    cv2.circle(bgr, (int(points[goal_j].pt[0]), int(points[goal_j].pt[1])), 10, (0,0,0), 5)
                else:
                    print("Nothing found")

    c = cv2.waitKey(1) & 0xFF

    keys1 = [ord(ki) for ki in "rftgyh"]
    keys2 = [ord(ki) for ki in "ujikol"]
    for k1 in range(len(keys1)):
        if c == keys1[k1]:
            dirrection = 1 if (k1 % 2)==0 else -1
            which = k1 // 2
            LBOUND[which] += dirrection
    for k2 in range(len(keys2)):
        if c == keys2[k2]:
            dirrection = 1 if (k2 % 2)==0 else -1
            which = k2 // 2
            UBOUND[which] += dirrection
    if c == ord('q'):
        break
    print(str(LBOUND) +' '+ str(UBOUND))

cap.release()
cv2.destroyAllWindows()
