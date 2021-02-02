import numpy as np

a = np.expand_dims(np.array([4,5,6,7,8,9,10]), axis=0)
print(a.shape)
a = np.repeat(a, 200, axis=0)
print(a.shape)