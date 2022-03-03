import matplotlib.pyplot as plt
import numpy as np
from gradient import *

def function(x):
    return (1/20)*x[0]**2 + x[1]**2

x = np.arange(-10.0, 11.0, 0.01)
y = np.arange(-5.0, 6.0, 0.01)
xx, yy = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

z = function(np.array([xx, yy]))
ax.plot_surface(xx, yy, z, cmap='terrain')
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()