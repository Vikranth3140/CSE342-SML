import numpy as np
from tqdm import tqdm

# Load the MNIST dataset from the local file
path = 'mnist.npz'  # Update this path with the local file path
with np.load(path) as data:
    x_train = data['x_train']
    y_train = data['y_train']

# Flatten the images to 1D arrays
x_train_flattened = x_train.reshape(x_train.shape[0], -1)

# Define the number of terminal nodes
num_nodes = 3

# Function to calculate Gini index
def calculate_gini(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini_index = 1 - np.sum(probabilities**2)
    return gini_index

# Function to find the best split for a given dimension
def find_best_split(feature_values, labels):
    min_gini = float('inf')
    best_split = None
    for value in np.unique(feature_values):
        left_indices = np.where(feature_values <= value)[0]
        right_indices = np.where(feature_values > value)[0]
        
        gini_left = calculate_gini(labels[left_indices])
        gini_right = calculate_gini(labels[right_indices])
        
        total_gini = (len(left_indices) * gini_left + len(right_indices) * gini_right) / len(labels)
        
        if total_gini < min_gini:
            min_gini = total_gini
            best_split = value
    
    return min_gini, best_split

# Initialize a decision tree structure
decision_tree = {}

# Grow the decision tree with 3 terminal nodes
for node in tqdm(range(num_nodes), desc='Growing Decision Tree'):
    best_dimension = None
    best_split_value = None
    min_gini_index = float('inf')
    
    for dimension in tqdm(range(x_train_flattened.shape[1]), desc=f'Node {node} Split'):
        feature_values = x_train_flattened[:, dimension]
        gini_index, split_value = find_best_split(feature_values, y_train)
        
        if gini_index < min_gini_index:
            min_gini_index = gini_index
            best_dimension = dimension
            best_split_value = split_value
    
    # Store the best split for this node
    decision_tree[node] = {'dimension': best_dimension, 'split_value': best_split_value}

    # Update labels for next iteration (considering the split)
    split_indices = np.where(x_train_flattened[:, best_dimension] <= best_split_value)[0]
    y_train[split_indices] = node  # Update labels to indicate node membership

# Print the decision tree structure
for node, details in decision_tree.items():
    print(f"Node {node}: Split on Dimension {details['dimension']} at Value {details['split_value']}")