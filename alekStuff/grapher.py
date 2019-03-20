
import json
import matplotlib.pyplot as plt
import numpy as np

with open("test.json", "r") as f:
    data = json.load(f)

with open("screenSize.json", "r") as f:
    screenSize = json.load(f)

data = np.array(data)
plt.scatter(data[:,0], data[:,1])
plt.scatter([0, screenSize[1]], [0,screenSize[0]])
plt.show()

