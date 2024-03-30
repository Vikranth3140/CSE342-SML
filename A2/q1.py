import numpy as np

# Loading the dataset
data = np.load("mnist.npz")

# Check the shapes of the loaded arrays
x_train = data['x_train']
y_train = data['y_train']
x_test = data["x_test"]
y_test = data["y_test"]

# Reshape images to make them 784-dimensional
x_train_flat = data["x_train"].reshape(-1, 784)
x_test_flat = data["x_test"].reshape(-1, 784)

# Filter data for classes 0, 1, and 2
indices_0 = np.where(y_train == 0)[0]
indices_1 = np.where(y_train == 1)[0]
indices_2 = np.where(y_train == 2)[0]

x_train_0 = x_train[indices_0]
x_train_1 = x_train[indices_1]
x_train_2 = x_train[indices_2]

# Combine data for classes 0, 1, and 2
x_train_combined = np.concatenate((x_train_0, x_train_1, x_train_2), axis=0)

# Center the data
X_mean = np.mean(x_train_combined, axis=0)
X_centered = x_train_combined - X_mean

# Calculate the covariance matrix
cov_matrix = np.dot(X_centered, X_centered.T) / 999

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select the top p eigenvectors
p = 10
U = sorted_eigenvectors[:, :p]

# Project the centered data onto the top p eigenvectors
Y = np.dot(U.T, X_centered)

# Reconstruct the data
X_recon = np.dot(U, Y)

# Now you can use X_recon for further analysis or tasks