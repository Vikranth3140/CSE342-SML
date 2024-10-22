import numpy as np
from tqdm import tqdm

path = 'mnist.npz'
with np.load(path) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

X_mean = np.mean(x_train_flattened, axis=0)
X_centered = x_train_flattened - X_mean
cov_matrix = np.dot(X_centered.T, X_centered) / len(x_train_flattened)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
U = eigenvectors[:, sorted_indices]
U_10 = U[:, :10]

Y_train = np.dot(U_10.T, X_centered.T).T

num_nodes = 3

def calculate_gini(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini_index = 1 - np.sum(probabilities**2)
    return gini_index

def find_best_split(feature_values, labels):
    min_gini = float('inf')
    best_split = None
    for value in np.sort(np.unique(feature_values)):
        left_indices = np.where(feature_values <= value)[0]
        right_indices = np.where(feature_values > value)[0]
        
        gini_left = calculate_gini(labels[left_indices])
        gini_right = calculate_gini(labels[right_indices])
        
        total_gini = (len(left_indices) * gini_left + len(right_indices) * gini_right) / len(labels)
        
        if total_gini < min_gini:
            min_gini = total_gini
            best_split = value
    
    return min_gini, best_split

decision_tree = {}

for node in tqdm(range(num_nodes), desc='Growing Decision Tree'):
    best_dimension = None
    best_split_value = None
    min_gini_index = float('inf')
    
    for dimension in tqdm(range(10), desc=f'Node {node} Split'):
        feature_values = Y_train[:, dimension]
        gini_index, split_value = find_best_split(feature_values, y_train)
        
        if gini_index < min_gini_index:
            min_gini_index = gini_index
            best_dimension = dimension
            best_split_value = split_value
    
    node_class = np.argmax(np.bincount(y_train))
    decision_tree[node] = {'dimension': best_dimension, 'split_value': best_split_value, 'class': node_class}

    split_indices = np.where(Y_train[:, best_dimension] <= best_split_value)[0]
    y_train[split_indices] = node_class

X_test_centered = x_test_flattened - X_mean
Y_test = np.dot(U_10.T, X_test_centered.T).T

predicted_classes = []
actual_classes = y_test.tolist()

def predict_class(sample, tree):
    node = 0
    while True:
        split_dim = tree[node]['dimension']
        split_value = tree[node]['split_value']
        if sample[split_dim] <= split_value:
            if isinstance(tree[node], dict) and 'left' in tree[node]:
                node = tree[node]['left']
        else:
            if isinstance(tree[node], dict) and 'right' in tree[node]:
                node = tree[node]['right']
            else:
                predicted_classes.append(tree[node]['class'])
                break

for sample in tqdm(Y_test.T, desc='Predicting Classes'):
    predict_class(sample, decision_tree)

correct_predictions = np.sum(np.array(predicted_classes) == np.array(actual_classes))
total_samples = len(y_test)
accuracy = correct_predictions / total_samples

class_wise_accuracy = {}
for cls in np.unique(y_test):
    cls_indices = np.where(y_test == cls)[0]
    correct_cls_predictions = np.sum(np.array(predicted_classes)[cls_indices] == cls)
    total_cls_samples = len(cls_indices)
    if total_cls_samples > 0:
        cls_accuracy = correct_cls_predictions / total_cls_samples
        class_wise_accuracy[cls] = cls_accuracy
    else:
        class_wise_accuracy[cls] = 0.0

print(f"Overall Accuracy: {accuracy:.4f}")
print("Class-wise Accuracy:")
for cls, cls_acc in class_wise_accuracy.items():
    print(f"Class {cls}: {cls_acc:.4f}")