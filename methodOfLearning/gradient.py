import numpy as np

def gradient(f, X):
    h = 1e-4
    grad = np.zeros_like(X)

    for idx, x in enumerate(X):
        grad_tmp = np.zeros_like(x)

        for i in range(x.size):
            temp = x[i]
            x[i] = float(temp) + h
            fxh1 = f(x)

            x[i] = float(temp) - h
            fxh2 = f(x)

            grad_tmp[i] = (fxh1 - fxh2) / (2*h)
            x[i] = temp
        grad[idx] = grad_tmp

    return grad


def gradient2(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        temp = x[i]
        x[i] = float(temp) + h
        fxh1 = f(x)

        x[i] = float(temp) - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = temp

    return grad