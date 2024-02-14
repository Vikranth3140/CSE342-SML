import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
mnist_data = np.load('../mnist.npz')

# Extract data
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']

# Create a figure
fig, axs = plt.subplots(10, 5, figsize=(10, 20))

# For each class
for i in range(10):
    # Get 5 samples from the class
    samples = x_train[y_train == i][:5]
    
    # For each sample
    for j in range(5):
        # Plot the sample
        axs[i, j].imshow(samples[j], cmap='gray')
        axs[i, j].axis('off')

plt.show()