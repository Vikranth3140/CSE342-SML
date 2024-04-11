import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

# Define the predict_class function
def predict_class(sample, tree):
    node = 0
    while True:
        split_dim = tree[node]['dimension']
        split_value = tree[node]['split_value']
        if sample[split_dim] <= split_value:
            if isinstance(tree[node], dict) and 'left' in tree[node]:
                node = tree[node]['left']
            else:
                return tree[node]['class']
        else:
            if isinstance(tree[node], dict) and 'right' in tree[node]:
                node = tree[node]['right']
            else:
                return tree[node]['class']

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

trees = []
tree_predictions = []

for i in range(5):
    Y_bagged, y_bagged = resample(Y_train, y_train, replace=True, random_state=i)
    trees.append(Y_bagged)

    predictions = []
    for sample in Y_test:
        predictions.append([predict_class(sample, tree) for tree in Y_bagged])
    tree_predictions.append(predictions)

def majority_vote(predictions):
    majority = np.zeros_like(predictions[0])
    for preds in predictions:
        for i in range(len(preds)):
            counts = np.bincount(preds[i])
            majority[i] += np.argmax(counts)
    majority /= len(predictions)
    return majority.astype(int)

majority_predictions = majority_vote(tree_predictions)

total_samples = len(y_test)
correct_predictions = np.sum(majority_predictions == y_test)
accuracy = correct_predictions / total_samples

class_wise_correct = np.zeros(10)
class_wise_total = np.zeros(10)
for i in range(total_samples):
    if majority_predictions[i] == y_test[i]:
        class_wise_correct[y_test[i]] += 1
    class_wise_total[y_test[i]] += 1

class_wise_accuracy = class_wise_correct / class_wise_total

print(f"Total Accuracy: {accuracy:.4f}")
print("Class-wise Accuracy:")
for cls, cls_acc in enumerate(class_wise_accuracy):
    print(f"Class {cls}: {cls_acc:.4f}")