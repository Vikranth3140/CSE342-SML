import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loading the dataset
data = np.load("mnist.npz")

# Extracting features and labels
x_train = data['x_train']
y_train = data['y_train']
x_test = data["x_test"]
y_test = data["y_test"]

# Filter samples for digits 0 and 1, and label them as -1 and 1
X_binary = []
y_binary = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        X_binary.append(x_train[i].flatten())  # Flatten the images to 1D array
        y_binary.append(-1)
    elif y_train[i] == 1:
        X_binary.append(x_train[i].flatten())
        y_binary.append(1)

X_binary = np.array(X_binary)
y_binary = np.array(y_binary)

# Split the dataset into train and validation sets, keeping 1000 samples from each class for validation
X_train, X_val, y_train, y_val = train_test_split(X_binary, y_binary, test_size=2000, stratify=y_binary, random_state=42)

# Reshape the training dataset
x_train_flat = X_train.reshape(-1, 784)

# Computing centralized PCA
x_train_centered = x_train_flat - np.mean(x_train_flat, axis=0)
num_of_samples = len(x_train_flat)
cov_matrix = np.dot(x_train_centered.T, x_train_centered) / (num_of_samples - 1)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sort_ind = np.argsort(eigenvalues)[::-1]
U = eigenvectors[:, sort_ind]

# Applying PCA to reduce the dimension to p=5
Up = U[:, :5]
Yp = np.dot(Up.T, x_train_centered.T)
X_train_pca = Yp.T

# Initialize variables for best split
best_feature = None
best_threshold = None
best_error = np.inf

# Iterate through each feature
for feature_idx in range(X_train_pca.shape[1]):
    # Get unique values in the feature and sort them
    unique_values = np.sort(np.unique(X_train_pca[:, feature_idx]))
    
    # Compute candidate split points as midpoints
    midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    
    # Evaluate split points
    for threshold in midpoints:
        # Make predictions based on the current split
        predictions = np.where(X_train_pca[:, feature_idx] <= threshold, -1, 1)
        
        # Compute weighted miss-classification error
        error = np.sum(predictions != y_train) / len(y_train)
        
        # Update best split if error is lower
        if error < best_error:
            best_feature = feature_idx
            best_threshold = threshold
            best_error = error

# Define the decision stump (h1(x)) based on the best split
def h1(x):
    return np.where(x[:, best_feature] <= best_threshold, -1, 1)

# Reshape the validation dataset
x_val_flat = X_val.reshape(-1, 784)

# Evaluate the performance of h1(x) on the validation set
X_val_centered = x_val_flat - np.mean(x_val_flat, axis=0)
Y_val_pca = np.dot(Up.T, X_val_centered.T)
X_val_pca = Y_val_pca.T
y_val_pred = h1(X_val_pca)
accuracy_h1 = accuracy_score(y_val, y_val_pred)
print(f"Accuracy of h1(x) on validation set: {accuracy_h1}")