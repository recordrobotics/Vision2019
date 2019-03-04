import cv2

for i in range(40, 200):
    left = cv2.imread("./images/left/" + str(i) + "img.jpg")
    right = cv2.imread("./images/right/" + str(i) + "img.jpg")

    concat = cv2.hconcat([ left, right ])

    cv2.imshow("Frame " + str(i), concat)
    while cv2.waitKey(0) & 0xFF != ord('n'):
        pass
    cv2.destroyAllWindows()
