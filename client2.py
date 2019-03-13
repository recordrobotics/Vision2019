from networktables import NetworkTables as nt

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while 1:
    t = sd.getNumber("gyro.yaw", -3.0)
    print(t)
