import numpy as np
batch = 1000000

dims = (84,84)
rand1 = np.empty((batch,4) + dims, dtype=np.uint8)
# rand2 = np.random.rand(batch,84,84,4)
# rand3 = np.random.rand(batch)
# rand4 = np.random.rand(batch)

print (batch,4)+dims
print rand1.shape
print rand1.nbytes
print rand1.itemsize


