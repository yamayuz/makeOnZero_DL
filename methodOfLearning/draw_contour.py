import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.01)
y = np.arange(-10, 10, 0.01)
X, Y = np.meshgrid(x, y)

z = (1/20)*X**2 + Y**2

cont = plt.contour(X, Y, z, levels=150)
plt.show()