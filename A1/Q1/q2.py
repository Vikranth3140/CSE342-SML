import numpy as np
import matplotlib.pyplot as plt

mnist_data = np.load('../mnist.npz')

x_train = mnist_data['x_train']
y_train = mnist_data['y_train']

fig, axs = plt.subplots(10, 5, figsize=(10, 20))

for i in range(10):
    samples = x_train[y_train == i][:5]
    
    for j in range(5):
        axs[i, j].imshow(samples[j], cmap='gray')
        axs[i, j].axis('off')

plt.show()