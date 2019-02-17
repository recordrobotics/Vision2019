import cv2
import time

left_cap = cv2.VideoCapture(1)
right_cap = cv2.VideoCapture(2)

i = 0
while True:
    left_cap.grab()
    right_cap.grab()

    l_s, l_img = left_cap.retrieve()
    r_s, r_img = right_cap.retrieve()

    if l_s and r_s:
        i += 1
        cv2.imwrite("./images/right/" + str(i) + "img.jpg", r_img)
        cv2.imwrite("./images/left/" + str(i) + "img.jpg", l_img)

        time.sleep(0.2)
