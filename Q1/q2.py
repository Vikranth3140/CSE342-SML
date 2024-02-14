import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Load MNIST data
mnist_data = np.load('../mnist.npz')

# Extract data
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']

# Vectorize the images
x_train = x_train.reshape((x_train.shape[0], -1))

# Create a QDA object
qda = QuadraticDiscriminantAnalysis(store_covariance=True)  # Set store_covariance to True

# Fit the QDA model
qda.fit(x_train, y_train)

# Print the mean vectors
print("Mean vectors:")
for i, mean_vec in enumerate(qda.means_):
    print(f"Class {i}: {mean_vec}")

# Print the covariance matrices
print("\nCovariance matrices:")
for i, cov_mat in enumerate(qda.covariance_):
    print(f"Class {i}: {cov_mat}")