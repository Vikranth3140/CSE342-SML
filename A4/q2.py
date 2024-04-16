import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

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
best_ssr = np.inf

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
        
        # Compute SSR for the predictions
        ssr = np.sum((predictions - y_train) ** 2)
        
        # Update best split if SSR is lower
        if ssr < best_ssr:
            best_feature = feature_idx
            best_threshold = threshold
            best_ssr = ssr

# Define the decision stump (h1(x)) based on the best split
def h1(x):
    return np.where(x[:, best_feature] <= best_threshold, -1, 1)

# Train the decision stump on the training data
decision_stump = DecisionTreeRegressor(max_depth=1, random_state=42)
decision_stump.fit(X_train_pca, y_train)

# Make predictions on the training set using the decision stump
y_train_pred = decision_stump.predict(X_train_pca)

# Calculate the Mean Squared Error (MSE) using the decision stump
mse_train = mean_squared_error(y_train, y_train_pred)

print(f"MSE on training set using decision stump h1(x): {mse_train}")

# Compute residue using y - 0.01 * h1(x)
residue = y_train - 0.01 * h1(X_train_pca)

# Calculate the Mean Squared Error (MSE) of the residue
mse_residue = np.mean(residue ** 2)

print(f"MSE of residue using y - 0.01 * h1(x): {mse_residue}")

# Compute residue using y - 0.01 * h1(x)
residue = y_train - 0.01 * h1(X_train_pca)

# Calculate the Mean Squared Error (MSE) of the residue
mse_residue = np.mean(residue ** 2)

print(f"MSE of residue using y - 0.01 * h1(x): {mse_residue}")

# Update labels based on negative gradients (absolute loss)
gradient = y_train - 0.01 * h1(X_train_pca)
# Update labels for the second decision stump
updated_y_train = gradient - 0.01 * h2(X_train_pca, updated_labels=True)

# Train the decision tree h2(x) on the training data with updated labels
decision_tree_h2 = DecisionTreeRegressor(max_depth=1, random_state=42)
decision_tree_h2.fit(X_train_pca, updated_y_train)

# Make predictions using h2(x)
h2_predictions = decision_tree_h2.predict(X_train_pca)

# Compute the residue using y - 0.01h1(x) - 0.01h2(x)
residue_h2 = y_train - 0.01 * h1(X_train_pca) - 0.01 * h2_predictions

# Calculate the Mean Squared Error (MSE) of the residue
mse_residue_h2 = np.mean(residue_h2 ** 2)

print(f"MSE of residue using y - 0.01h1(x) - 0.01h2(x): {mse_residue_h2}")

# Initialize lists to store alpha values, stump predictions, and residues
alphas = []
stump_preds = []
residues = []

# Initialize weights for AdaBoost.M1
weights_train = np.ones(len(y_train)) / len(y_train)

# Train decision stumps sequentially
for i in range(300):
    # Compute the negative gradient
    gradient = y_train - 0.01 * np.dot(np.array(stump_preds).T, np.array(alphas))
    
    # Update labels for the current iteration
    updated_y_train = gradient
    
    # Initialize Decision Tree Regressor for the stump
    stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    
    # Train the stump on the training data with updated labels and weights
    stump.fit(X_train_pca, updated_y_train, sample_weight=weights_train)
    
    # Make predictions on the training set
    stump_pred_train = stump.predict(X_train_pca)
    
    # Compute weighted error
    weighted_error = np.sum(weights_train * (stump_pred_train != updated_y_train)) / np.sum(weights_train)
    
    # Compute alpha for the stump
    alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
    alphas.append(alpha)
    
    # Update weights for the next iteration
    weights_train *= np.exp(-alpha * updated_y_train * stump_pred_train)
    weights_train /= np.sum(weights_train)  # Normalize weights
    
    # Store the predictions of the stump
    stump_preds.append(stump_pred_train)
    
    # Compute the residue for this iteration
    residue = y_train - 0.01 * np.dot(np.array(stump_preds).T, np.array(alphas))
    residues.append(residue)

# Calculate the Mean Squared Error (MSE) of the residue
mse_residue = np.mean(np.array(residues) ** 2)

print(f"MSE of residue using updated labels: {mse_residue}")

# Initialize lists to store MSE values on the validation set and corresponding number of trees
mse_val_list = []
num_trees_list = []

# Train decision stumps sequentially and compute MSE on validation set after each iteration
for i in range(300):
    # Compute the negative gradient
    gradient = y_train - 0.01 * np.dot(np.array(stump_preds).T, np.array(alphas))
    
    # Update labels for the current iteration
    updated_y_train = gradient
    
    # Initialize Decision Tree Regressor for the stump
    stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    
    # Train the stump on the training data with updated labels and weights
    stump.fit(X_train_pca, updated_y_train, sample_weight=weights_train)
    
    # Make predictions on the validation set
    stump_pred_val = stump.predict(X_val_pca)
    
    # Compute MSE on validation set
    mse_val = mean_squared_error(y_val, stump_pred_val)
    
    # Store MSE and number of trees for plotting
    mse_val_list.append(mse_val)
    num_trees_list.append(i + 1)  # Number of trees starts from 1
    
    # Update weights for the next iteration
    weights_train *= np.exp(-alpha * updated_y_train * stump_pred_train)
    weights_train /= np.sum(weights_train)  # Normalize weights

# Plotting MSE vs. number of trees
plt.figure(figsize=(10, 6))
plt.plot(num_trees_list, mse_val_list, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('MSE on Validation Set')
plt.title('MSE vs. Number of Trees')
plt.grid(True)
plt.show()

# Find the tree with the lowest MSE on validation set
best_tree_idx = np.argmin(mse_val_list)
best_tree = DecisionTreeRegressor(max_depth=1, random_state=42)
best_tree.fit(X_train_pca, y_train)  # Train the best tree on the full training set

# Make predictions on the test set using the best tree
y_test_pred = best_tree.predict(X_test_pca)

# Compute MSE on the test set using the best tree
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Lowest MSE on validation set: {mse_val_list[best_tree_idx]} (Tree {best_tree_idx + 1})")
print(f"MSE on test set using the best tree: {mse_test}")