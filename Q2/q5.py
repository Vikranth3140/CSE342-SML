import numpy as np
import matplotlib.pyplot as plt
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

Y = np.concatenate(samples, axis=0)

X = Y.T

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

# Define the values of p
p_values = [5, 10, 20]

# For each value of p
for p in p_values:
    # Select the first p eigenvectors from U
    U_p = U[:, :p]

    # Perform the PCA transformation
    Y_p = np.dot(U_p.T, X.T)

    # Reconstruct the data
    X_recon_p = np.dot(U_p, Y_p)

    # Add the mean that was removed from X
    X_recon_p = X_recon_p + mean_X

    # Reshape each column to 28x28
    X_recon_p = X_recon_p.reshape((-1, 28, 28))

    # Create a figure
    fig, axs = plt.subplots(10, 5, figsize=(10, 20))

    # For each class
    for i in range(10):
        # Get 5 samples from the class
        samples = X_recon_p[y_train[:1000] == i][:5]
        
        # For each sample
        for j in range(5):
            # Plot the sample
            axs[i, j].imshow(samples[j], cmap='gray')
            axs[i, j].axis('off')

    plt.suptitle(f"Reconstructed images with p={p}")
    plt.show()