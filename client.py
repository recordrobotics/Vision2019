from networktables import NetworkTables as nt
import cv2
import time
import numpy as np
import pickle
import yaml

import time

C = 1.0


nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

d = pickle.load(open("pmtxs.data", "rb"))
d2 = yaml.load(open("stereo_calib", "r"))

print(d2)

l_pmtx = np.array(d2["P1"]["data"]).reshape(3, 4) #d["lp"]
r_pmtx = np.array(d2["P2"]["data"]).reshape(3, 4) #d["rp"]
l_rmtx = np.array(d2["R1"]["data"]).reshape(3, 3) #d["lr"]
r_rmtx = np.array(d2["R2"]["data"]).reshape(3, 3) #d["lr"]
r_mtx = np.array(d2["K2"]["data"]).reshape(3, 3) #d["lr"]
l_mtx = np.array(d2["K1"]["data"]).reshape(3, 3) #d["lr"]
r_dist = np.array(d2["D2"]["data"]) #d["lr"]
l_dist = np.array(d2["D1"]["data"]) #d["lr"]
t = np.array(d2["T"]) #d["lr"]
r = np.array(d2["R"]["data"]).reshape(3, 3) #d["lr"]

print(t)


while 1:
    t1 = sd.getNumber("tapes|PI_1", -3.0)
    t2 = sd.getNumber("tapes|PI_2", -3.0)

    print(t1)
    if t1 >= -1.0 and t2 >= -1.0:
        print(0.5 * (t1 + t2))
"""
        l_test_points = np.array([[[t1, 240]]])
        r_test_points = np.array([[[t2, 240]]])

        start = time.time()

        l_undistorted = cv2.undistortPoints(l_test_points, l_mtx, l_dist, None, l_rmtx, l_pmtx)
        r_undistorted = cv2.undistortPoints(r_test_points, r_mtx, r_dist, None, r_rmtx, r_pmtx)

        print("\nleft undistorted: " + str(l_undistorted))
        print("\nright undistorted: " + str(r_undistorted))

        projected_points = cv2.triangulatePoints(l_pmtx, r_pmtx, l_test_points, r_test_points)
        projected_points2 = cv2.triangulatePoints(l_pmtx, r_pmtx, l_undistorted, r_undistorted)

        end = time.time()

        #print("\nElapsed: " + str(end - start))

        #print("\nProjected: " + str((projected_points[:3] / projected_points[3]).T))
        print("\nProjected2: " + str((projected_points2[:3] / projected_points2[3]).T))

        time.sleep(0.01)
        """
