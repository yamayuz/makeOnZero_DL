import numpy as np
import matplotlib.pyplot as plt
from gradient import *
import copy

def f(x):
    return (1/20)*x[0]**2 + x[1]**2

class AdsGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, grad, x):
        if self.h is None:
            self.h = np.zeros_like(x)

        self.h += grad**2
        x -= self.lr * (1/(np.sqrt(self.h) + 1e-7)) * grad


"""変数定義"""
init_pos = (-7.0, 5.0)
x = np.array([init_pos[0], init_pos[1]])
h = np.array([0, 0])
pos_history = []
optimizers = AdsGrad(lr=1.5)

"""補正"""
for i in range(30):
    pos_history.append(copy.deepcopy(x))
    grad = gradient2(f, x)
    optimizers.update(grad, x)

x_history = np.array(pos_history)   

"""グラフ描画"""
x0 = np.arange(-10, 10, 0.1)
x1 = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(x0, x1)
Z = f(np.array([X, Y]))

plt.plot(x_history.T[0], x_history.T[1])
plt.contour(X, Y, Z, levels=20, colors='gainsboro')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("AdaGrad")
plt.show()