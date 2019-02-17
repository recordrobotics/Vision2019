import cv2
import time
import numpy as np
import pickle

l_test_points = np.array([[-100], [3540]], dtype=np.float32)
r_test_points = np.array([[100], [5040]], dtype=np.float32)

d = pickle.load(open("pmtxs.data", "rb"))

l_pmtx = d["l"]
r_pmtx = d["r"]

start = time.time()
projected_points = cv2.triangulatePoints(l_pmtx, r_pmtx, l_test_points, r_test_points)
end = time.time()

print("Elapsed: " + str(end - start))

print(projected_points)
