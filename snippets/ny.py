import numpy as np

a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)  # + is broadcast

print(np.arange(6))
print(np.arange(0, 60, 10))
print(np.arange(0, 60, 10).reshape((-1, 1)))
print(a)


L = [1, "2", True]  # every element is object
print(L)

a = np.array(L)
print(a)

