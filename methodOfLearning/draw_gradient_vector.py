import matplotlib.pyplot as plt
import numpy as np
from gradient import *

def function(x):
    return (1/20)*x[0]**2 + x[1]**2

x = np.arange(-10.0, 11.0, 1)
y = np.arange(-5.0, 6.0, 1)
xx, yy = np.meshgrid(x, y)

x0 = xx.flatten()
x1 = yy.flatten()
X = np.array([x0, x1]).T

grad = gradient(function, X)

fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot(111)

ax.grid()
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 5)

ax.quiver(x0, x1, -grad.T[0], -grad.T[1], color = "red", angles = 'xy')
plt.show()
