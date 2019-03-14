
import numpy as np

data = np.array([[1,2],[3,6],[6,12],[4,8],[1,2.4],[10,19],[12,22]])
coeff = np.polyfit(data[:,0], data[:,1], 1)
print(coeff)

import matplotlib.pyplot as plt

plt.scatter(data[:,0],data[:,1])
plt.plot([min(data[:,0]), max(data[:,0])], [min(data[:,0])*coeff[0]+coeff[1], max(data[:,0])*coeff[0]+coeff[1]])
plt.show()

