import numpy as np
from sklearn.metrics import mean_squared_error

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

# Compute the mean of X
mean_X = np.mean(X, axis=0)

# Subtract the mean from X
X = X - mean_X

# Compute the covariance matrix
S = np.cov(X, rowvar=False)

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(S)

# Sort the eigenvalues in descending order, and get the indices
indices = np.argsort(eigenvalues)[::-1]

# Sort the eigenvalues and eigenvectors
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

# Create the matrix U
U = eigenvectors

# Perform the PCA transformation
Y = np.dot(U.T, X.T)

# Reconstruct the data
X_recon = np.dot(U, Y)

# Compute the MSE between X and X_recon
mse = mean_squared_error(X, X_recon.T)

print(f"MSE between X and X_recon: {mse}")