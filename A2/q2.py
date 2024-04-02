import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tqdm import tqdm

# Load the MNIST dataset from the local file
path = 'mnist.npz'  # Update this path with the local file path
with np.load(path) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# Flatten the images to 1D arrays
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

# Initialize lists to store the decision trees and their predictions
trees = []
tree_predictions = []

# Bagging: Create 5 different datasets and train decision trees
for i in tqdm(range(5), desc='Bagging and Training Decision Trees'):
    # Sample with replacement to create a new dataset
    x_bagged_flat, y_bagged = resample(x_train_flattened, y_train, replace=True, random_state=i)

    # Train a decision tree on the bagged dataset
    tree = DecisionTreeClassifier(max_depth=None, random_state=i)
    tree.fit(x_bagged_flat, y_bagged)
    
    # Store the trained tree
    trees.append(tree)

    # Make predictions on the test set and store them for majority voting
    tree_predictions.append(tree.predict(x_test_flattened))

# Perform majority voting among the tree predictions
def majority_vote(predictions):
    # Initialize an array to store the majority voting predictions
    majority = np.zeros(predictions[0].shape)
    
    # Perform majority voting for each sample
    for i in range(len(predictions[0])):
        counts = np.bincount([preds[i] for preds in predictions])
        majority[i] = np.argmax(counts)
    
    return majority.astype(int)

# Get the majority voting predictions
majority_predictions = majority_vote(tree_predictions)

# Calculate accuracy and class-wise accuracy
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