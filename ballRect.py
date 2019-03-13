import cv2
import numpy as np
import time
from networktables import NetworkTables as nt


frameScalingFactor = 0.5

# PARAMS
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100000#*frameScalingFactor*frameScalingFactor
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


params_tape = cv2.SimpleBlobDetector_Params()
params_tape.filterByArea = True
params_tape.minArea = 600#*frameScalingFactor*frameScalingFactor
params_tape.maxArea = 1000000#*frameScalingFactor*frameScalingFactor
params_tape.filterByCircularity = False
params_tape.filterByColor = False
params_tape.blobColor = 255
params_tape.filterByConvexity = False
params_tape.minConvexity = 0#0.3
params_tape.maxConvexity = 1.0
params_tape.filterByInertia = False
params_tape.minInertiaRatio = 0#0.04
params_tape.maxInertiaRatio = 1#0.5
detector_tape = cv2.SimpleBlobDetector_create(params_tape)

LBOUND_ORANGE = np.array([0, 100, 200])
UBOUND_ORANGE = np.array([50, 255, 255])

LBOUND_BRIGHT = np.array([0, 0, 250])
UBOUND_BRIGHT = np.array([60, 100, 255])

LBOUND_WHITE_BGR = np.array([254, 254, 254])
UBOUND_WHITE_BGR = np.array([255, 255, 255])

LBOUND_GREEN = np.array([45, 0, 250])
UBOUND_GREEN = np.array([75, 100, 255])
# END PARAMS

print("OpenCV version: " + cv2.__version__)

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

def kernel(bgr):
    bgr = cv2.blur(bgr, (11, 11))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_tape = cv2.inRange(hsv, LBOUND_GREEN, UBOUND_GREEN)
    mask_tape = cv2.morphologyEx(mask_tape, cv2.MORPH_CLOSE, (7, 7), iterations=2)
    points_tape_maybe = detector_tape.detect(mask_tape)
    cv2.imshow("Tape mask", mask_tape)

    mask_white = cv2.inRange(bgr, LBOUND_WHITE_BGR,UBOUND_WHITE_BGR)
    mask_bright = cv2.inRange(hsv, LBOUND_BRIGHT, UBOUND_BRIGHT)
    mask_orange = cv2.inRange(hsv, LBOUND_ORANGE, UBOUND_ORANGE)
    mask = cv2.bitwise_or(mask_orange, mask_bright)
    mask = cv2.bitwise_and(cv2.bitwise_not(mask_white),mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (7, 7), iterations=2)
    points_ball = detector.detect(mask)
    cv2.imshow("Ball mask", mask)
    
    return points_ball, points_tape_maybe


print("SKETCHY")
# try:
cap = cv2.VideoCapture(0)
# except:
    # cap = cv2.VideoCapture(0)

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

        x = -1.0
        if len(points_tape) > 0:
        #     min_i, min_j = find_tapes(points_tape)
        #     if min_i >= 0:
        #         display_tape = cv2.drawKeypoints(bgr, [ points_tape[min_i], points_tape[min_j] ], np.array([]), (0, 0, 255),
        #                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #         x = 0.5 * (points_tape[min_i].pt[0] + points_tape[min_j].pt[0])
        #         # x = (x - 720 * frameScalingFactor) / (720 * frameScalingFactor)
        #         cv2.imshow("yayasdjfklasjdf", display_tape)

            display_tape = cv2.drawKeypoints(bgr, points_tape, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Tape view", display_tape)

        best_idx = 0
        biggest_r = 0
        if len(points_ball) > 0:
            for p in range(len(points_ball)):
                current_r = points_ball[p].size
                if current_r > biggest_r:
                    biggest_r = current_r
                    best_idx = p

            display_ball = cv2.drawKeypoints(bgr, [points_ball[best_idx]], np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Ball view", display_ball)
            x = (points_ball[best_idx].pt[0] - bgr.shape[1] * 0.5) / (bgr.shape[1] * 0.5)
            y = points_ball[best_idx].pt[1] / bgr.shape[0]

        # print("x: " + str(x) + "  y: " + str(y))
        # sd.putNumber("ball_x|PI_2", x)
        # sd.putNumber("ball_y|PI_2", y)

    bound = LBOUND_GREEN
    bound2 = UBOUND_GREEN

    c = cv2.waitKey(1) & 0xFF
    if c == ord('r'):
        bound[0] += 1
    elif c == ord('f'):
        bound[0] -= 1
    elif c == ord('t'):
        bound[1] += 1
    elif c == ord('g'):
        bound[1] -= 1
    elif c == ord('y'):
        bound[2] += 1
    elif c == ord('h'):
        bound[2] -= 1
    if c == ord('u'):
        bound2[0] += 1
    elif c == ord('j'):
        bound2[0] -= 1
    elif c == ord('i'):
        bound2[1] += 1
    elif c == ord('k'):
        bound2[1] -= 1
    elif c == ord('o'):
        bound2[2] += 1
    elif c == ord('l'):
        bound2[2] -= 1
    elif c == ord('q'):
        break

    print(str(bound) +' '+ str(bound2))

cap.release()
cv2.destroyAllWindows()
