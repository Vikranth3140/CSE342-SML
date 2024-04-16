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

def h2(X_train_pca, weights_train):
    # Define the decision stump function h2(x) based on the best split
    def stump_predict(X):
        return np.where(X[:, best_feature] <= best_threshold, -1, 1)

    # Compute predictions based on the current split
    predictions = stump_predict(X_train_pca)

    return predictions

# Reshape the validation dataset
x_val_flat = X_val.reshape(-1, 784)

# Evaluate the performance of h1(x) on the validation set
X_val_centered = x_val_flat - np.mean(x_val_flat, axis=0)
Y_val_pca = np.dot(Up.T, X_val_centered.T)
X_val_pca = Y_val_pca.T
y_val_pred = h1(X_val_pca)
accuracy_h1 = accuracy_score(y_val, y_val_pred)
print(f"Accuracy of h1(x) on validation set: {accuracy_h1}")


# Initialize weights for AdaBoost.M1
weights_train = np.ones(len(y_train)) / len(y_train)

# Train the first decision stump h1(x) using the train set with the initial weights
predictions_h1 = h1(X_train_pca)

# Compute the error of the first decision stump h1(x)
error_h1 = np.sum(weights_train * (predictions_h1 != y_train)) / np.sum(weights_train)

# Compute alpha_1 using the error
alpha_1 = 0.5 * np.log((1 - error_h1) / error_h1)

# Update the weights
weights_train *= np.exp(-alpha_1 * y_train * predictions_h1)
weights_train /= np.sum(weights_train)  # Normalize the weights to sum up to 1


num_stumps = 300  # Total number of stumps to train
alphas = []  # List to store the alphas for each stump
accuracies_val = []  # List to store accuracies on the validation set
best_accuracy = 0.0
best_stump_idx = 0

# Initialize weights for AdaBoost.M1
weights_train = np.ones(len(y_train)) / len(y_train)

for i in range(num_stumps):
    # Train a decision stump using the train set with the updated weights
    predictions_stump = h2(X_train_pca, weights_train)  # Replace h2 with your decision stump function
    error_stump = np.sum(weights_train * (predictions_stump != y_train)) / np.sum(weights_train)
    alpha_stump = 0.5 * np.log((1 - error_stump) / error_stump)
    
    # Store alpha for this stump
    alphas.append(alpha_stump)

    # Update the weights for the next iteration
    weights_train *= np.exp(-alpha_stump * y_train * predictions_stump)
    weights_train /= np.sum(weights_train)  # Normalize the weights to sum up to 1
    
    # Evaluate accuracy on the validation set
    y_val_pred = np.sign(np.dot(X_val_pca, np.array(alphas)))  # Combine predictions of all stumps
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracies_val.append(accuracy_val)
    
    # Update best accuracy and best stump index
    if accuracy_val > best_accuracy:
        best_accuracy = accuracy_val
        best_stump_idx = i + 1  # Stump index starts from 1
    
    print(f"Iteration {i + 1}: Accuracy on validation set = {accuracy_val}")

print(f"\nBest accuracy on validation set: {best_accuracy} at iteration {best_stump_idx}")

# Plot accuracy on validation set vs. number of trees
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_stumps + 1), accuracies_val, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy on Validation Set')
plt.title('Accuracy on Validation Set vs. Number of Trees')
plt.grid(True)
plt.show()

# Evaluate the best stump on the test set
best_stump = h2(X_train_pca, weights_train[:best_stump_idx])
# Evaluate accuracy on the validation set
y_val_pred = np.sign(np.dot(X_val_pca, np.array(alphas).reshape(-1, 1)))  # Reshape alphas to align with dot product

accuracy_test = accuracy_score(y_test, y_val_pred)
print(f"\nAccuracy on test set using the best stump: {accuracy_test}")