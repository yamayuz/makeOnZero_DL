from twoLayerNet import TwoLayerNet
import numpy as np
from typing import OrderedDict


input_size = 2
hidden_size = 3
output_size = 2

x = np.array([1, 2])

twoLayerNet = TwoLayerNet(input_size, hidden_size, output_size)

# twoLayerNet.predict(x)

a = [1, 2, 3]

print(a)
a.reverse()
print(a)