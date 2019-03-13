from networktables import NetworkTables as nt

nt.initialize(server="roborio-6731-frc.local")
sd = nt.getTable("SmartDashboard")

while True:
    v0 = sd.getNumber("velocity0", -2.0)
    v1 = sd.getNumber("velocity1", -2.0)
    theta = sd.getNumber("theta", 0.0)
    l = sd.getNumber("drivetrain.left_encoder", -2.0)
    r = sd.getNumber("drivetrain.right_encoder", -2.0)

    print("v0: " + str(v0) + ", v1: " + str(v1) + ", theta: " + str(theta) + ", l: " + str(l) + ", r: " + str(r))
