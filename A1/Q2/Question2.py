import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# Loading the dataset
data = np.load("../mnist.npz")

# Check the shapes of the loaded arrays
x_train = data['x_train']
y_train = data['y_train']
x_test = data["x_test"]
y_test = data["y_test"]

# Reshape images to make them 784-dimensional
x_train_flat = data["x_train"].reshape(-1, 784)
x_test_flat = data["x_test"].reshape(-1, 784)

# Finding X 
X = np.zeros((784, 1000))
Y = np.zeros((1000,))
col_count = 0
for digit in range(10):
    digit_indices = np.where(y_train == digit)[0]
    np.random.shuffle(digit_indices)
    get_index = digit_indices[:100]
    Y[col_count:col_count+100] = y_train[get_index]  # Assign labels to Y
    selected_samples = x_train_flat[get_index]
    X[:, col_count:col_count+100] = selected_samples.T
    col_count += 100

# Print the shape of the data matrix X
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

# Remove mean from X.
X_centered = X - np.mean(X, axis=0)

# Computing covariance ,eigenvectors and eigenvalue. Applying PCA on the centralized PCA.
cov_matrix = np.dot(X_centered, X_centered.T) / 999
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1] #Sorting in descending order
sorted_eigenvalues = eigenvalues[sorted_indices]
U = eigenvectors[:, sorted_indices]
Y = np.dot(U.T, X_centered)
Xrecon = np.dot(U, Y)

# Compute the Mean Squared Error (MSE)
MSE = np.mean((np.real(X_centered) - np.real(Xrecon))**2)
print("MSE between X_centered and X_recon:", MSE)

# arr = [5, 10, 200, 784]
arr = [5, 10, 15]
for p in arr:
    Up = U[:, :p]
    Yp = np.dot(Up.T, X_centered)
    Xrecon_p = np.dot(Up, Yp)
    Xrecon_p += np.mean(X, axis=0)
    Xrecon_p = np.real(Xrecon_p)

    # Plot 5 images from each class
    fig, axes = plt.subplots(10, 5, figsize=(12, 10))
    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0][:5]
        for i, index in enumerate(digit_indices):
            axes[digit, i].imshow(Xrecon_p[:,(digit*100)+i].reshape(28, 28), cmap='gray')
            axes[digit, i].set_title(f"Digit {digit} (p={p})")
            axes[digit, i].axis('off')

    plt.tight_layout()
    plt.show()

# Pre-processing
class_cov_X = {}
class_mean_X = {}
col_count = 0
for digit in range(10):
    X_class = X[:, col_count:col_count+100]
    expected_val = np.mean(X_class,axis=0)
    class_mean_X[digit] = expected_val
    cov_matrix = np.dot(X_class, X_class.T) / 999
    cov_matrix += 1e-5 * np.identity(cov_matrix.shape[0]) 
    class_cov_X[digit] = cov_matrix
    col_count += 100


# Finding Determinat and Inverse of Covariance Matrix for computation
determinants = []
inverses = []
subset_class_priors = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
for k in range(10):
    determinants.append(np.linalg.slogdet(class_cov_X[k])[1])
    inverses.append(np.linalg.inv(class_cov_X[k]))