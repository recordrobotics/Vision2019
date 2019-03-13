
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

# Vassilios parameters: from before we messed with stuff
# PARAMS
# params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.minArea = 60#*frameScalingFactor*frameScalingFactor
# params.maxArea = 6000#*frameScalingFactor*frameScalingFactor
# params.filterByCircularity = False
# params.filterByColor = False
# params.blobColor = 255
# params.filterByConvexity = True
# params.minConvexity = 0.3
# params.maxConvexity = 1.0
# params.filterByInertia = True
# params.minInertiaRatio = 0.04
# params.maxInertiaRatio = 0.5
# detector = cv2.SimpleBlobDetector_create(params)

params_tape = cv2.SimpleBlobDetector_Params()
params_tape.filterByArea = True
params_tape.minArea = 600#*frameScalingFactor*frameScalingFactor
params_tape.maxArea = 60000#*frameScalingFactor*frameScalingFactor
params_tape.filterByCircularity = False
params_tape.filterByColor = False
params_tape.blobColor = 255
params_tape.filterByConvexity = False
params_tape.minConvexity = 0.3
params_tape.maxConvexity = 1.0
params_tape.filterByInertia = False
params_tape.minInertiaRatio = 0.04
params_tape.maxInertiaRatio = 0.5
detector_tape = cv2.SimpleBlobDetector_create(params_tape)

LBOUND_ORANGE = np.array([0, 100, 200])
UBOUND_ORANGE = np.array([50, 255, 255])

LBOUND_BRIGHT = np.array([0, 0, 250])
UBOUND_BRIGHT = np.array([60, 100, 255])

LBOUND_WHITE_BGR = np.array([254, 254, 254])
UBOUND_WHITE_BGR = np.array([255, 255, 255])

LBOUND_GREEN = np.array([90, 20, 200])
UBOUND_GREEN = np.array([110, 200, 255])

desired_x = -0.0122
desired_y = 0.4

# END PARAMS

# ALEK STUFF
gamma = 0.9
tapeAvgX = 0

ballGamma = 0.9
ballAvgX = 0
ballAvgY = 0

print("OpenCV version: " + cv2.__version__)
def find_tapes(points):
    min_i = -1
    min_j = -1
    min_score = 100
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

def kernel(bgr):
    bgr = cv2.blur(bgr, (7, 7))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_tape = cv2.inRange(hsv, LBOUND_GREEN, UBOUND_GREEN)
    #mask_tape = cv2.morphologyEx(mask_tape, cv2.MORPH_CLOSE, (19, 19), iterations=10)
    points_tape_maybe = detector_tape.detect(mask_tape)
    cv2.imshow("Tape mask", mask_tape)

    mask_white = cv2.inRange(bgr, LBOUND_WHITE_BGR,UBOUND_WHITE_BGR)
    mask_bright = cv2.inRange(hsv, LBOUND_BRIGHT, UBOUND_BRIGHT)
    mask_orange = cv2.inRange(hsv, LBOUND_ORANGE, UBOUND_ORANGE)
    mask = cv2.bitwise_or(mask_orange, mask_bright)
    mask = cv2.bitwise_and(cv2.bitwise_not(mask_white),mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (31, 31), iterations=2)
    points_ball = detector.detect(mask)

    # cv2.circle(bgr, (hsv.shape[1]//2, hsv.shape[0]//2), 40, (0,0,0), 10)
    # cv2.circle(bgr, (hsv.shape[0]//2, hsv.shape[1]//2), 10, (0,0,0), 5)
    cv2.imshow("Ball mask", mask)

    return points_ball, points_tape_maybe


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
        points_ball, points_tape = kernel(bgr)
        end = time.time()

        ### END MAIN LOOP
#        print("FPS: " + str(1.0 / (end - start)))

        tape_x = -1.0
        tape_y = -1.0
        if len(points_tape) > 1:

            # Display only the best
            min_i, min_j = find_tapes(points_tape)
            if min_i >= 0:
                display_tape = cv2.drawKeypoints(bgr, [ points_tape[min_i], points_tape[min_j] ], np.array([]), (0, 0, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                tape_x = 0.5 * (points_tape[min_i].pt[0] + points_tape[min_j].pt[0])
                tape_y = 0.5 * (points_tape[min_i].pt[1] + points_tape[min_j].pt[1])
                tape_x = (tape_x - bgr.shape[1] * 0.5) / (bgr.shape[1] * 0.5)
                tape_y = tape_y / bgr.shape[0]
                tapeAvgX = tapeAvgX*gamma+(1-gamma)*tape_x
                # look at what super Avg X looks like plz
                cv2.circle(display_tape, (int(tapeAvgX), bgr.shape[0]//2), 10, (255,255,0), 5)
                # x = (x - 720 * frameScalingFactor) / (720 * frameScalingFactor)
                cv2.imshow("display_tape", display_tape)

            # Display everything:
            # display_tape = cv2.drawKeypoints(bgr, points_tape, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("Tape view", display_tape)

        best_idx = 0
        biggest_r = 0
        ball_x = -1.0
        ball_y = -1.0
        if len(points_ball) > 0:
            for p in range(len(points_ball)):
                current_r = points_ball[p].size
                if current_r > biggest_r:
                    biggest_r = current_r
                    best_idx = p

            display_ball = cv2.drawKeypoints(bgr, [points_ball[best_idx]], np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            ball_x = (points_ball[best_idx].pt[0] - bgr.shape[1] * 0.5) / (bgr.shape[1] * 0.5)
            ball_y = points_ball[best_idx].pt[1] / bgr.shape[0]
            
            ballAvgX = ballGamma*ballAvgX + (1-ballGamma)*ball_x
            ballAvgY = ballGamma*ballAvgY + (1-ballGamma)*ball_y

            cv2.circle(display_ball, (int(ballAvgX), int(ballAvgY)), 15, (0,250,0), 10)
            cv2.imshow("Ball view", display_ball)


        if ball_x != -1.0 and ball_y != -1.0 and tape_x != -1.0 and tape_y != -1.0:
            x = ball_x - tape_x - desired_x
            y = ball_y - desired_y
            print("x: " + str(x) + "  y: " + str(y))
            sd.putNumber("ball_x|PI_2", x)
            sd.putNumber("ball_y|PI_2", y)
        else:
            sd.putNumber("ball_x|PI_2", -2.0)
            sd.putNumber("ball_y|PI_2", -2.0)

    bound1 = LBOUND_GREEN
    bound2 = UBOUND_GREEN

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
