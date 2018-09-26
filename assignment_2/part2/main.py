import numpy as np
import matplotlib.pyplot as plt

T = 35
tau = 10.0
s = 2.0
r = 1

ys = []
for t in range(T):
    phi = np.mod((t - s), tau) / tau
    y = 0

    if phi < 0.5*r:
        y = 2*phi / r
    elif 0.5*r < phi < r:
        y = 2 - 2*phi / r
    ys.append(y)

plt.scatter(range(T), ys)
plt.show()