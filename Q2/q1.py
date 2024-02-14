import numpy as np

# Load MNIST data
mnist_data = np.load('../mnist.npz')

# Extract data
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']

# Vectorize the images
x_train = x_train.reshape((x_train.shape[0], -1))

# Initialize an empty list to store the samples
samples = []

# For each class
for i in range(10):
    # Get the first 100 samples from the class
    class_samples = x_train[y_train == i][:100]
    
    # Add the samples to the list
    samples.append(class_samples)

# Concatenate the samples to form a 784x1000 matrix
X = np.concatenate(samples, axis=0)

print("Data matrix X created with shape:", X.shape)
print(X)