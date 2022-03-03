import numpy as np
import matplotlib.pyplot as plt
from gradient import *
import copy

def f(x):
    return (1/20)*x[0]**2 + x[1]**2

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.v = None
        self.m = None

    def update(self, grad, x):
        if self.v is None:
            self.v = np.zeros_like(x)
            self.m = np.zeros_like(x)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)  

        # self.m = (1/(1-self.beta1)) * (self.beta1*self.m + (1-self.beta1)*grad)
        # self.v = (1/(1-self.beta2)) * (self.beta2*self.v + (1-self.beta2)*grad**2)
        # x = x - lr_t * (self.m / (np.sqrt(self.v) + 1e-8))
        self.m += (1 - self.beta1) * (grad - self.m)
        self.v += (1 - self.beta2) * (grad**2 - self.v)
        x -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)


"""変数定義"""
init_pos = (-7.0, 5.0)
x = np.array([init_pos[0], init_pos[1]])
pos_history = []
optimizers = Adam(lr=0.3)

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
plt.title("Adam")
plt.show()