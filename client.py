from networktables import NetworkTables as nt
import time

C = 1.0


nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    t1 = sd.getNumber("tapes|PI_1", -3.0)
    t2 = sd.getNumber("tapes|PI_2", -3.0)

    d = -2.0
    if t1 != -3.0 and t2 != -3.0:
        if t1 == t2:
            d = -1.0
        else:
            d = C / (t2 - t1)

    print(d)

    time.sleep(0.01)
