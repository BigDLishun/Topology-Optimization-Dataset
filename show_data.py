import numpy as np
import matplotlib.pyplot as plt

path1 = 'random_64_128/11083.npz'
data = np.load(path1)
data = data['arr_0']
print (data.shape)
plt.imshow(data[0],cmap='Greys')
plt.show()
plt.imshow(data[-1],cmap='Greys')
plt.show()


