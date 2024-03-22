import numpy as np
import matplotlib.pyplot as plt

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

# Load the test set and reshape it
X = np.zeros((784, 1000))
Y_train= np.zeros((1000,))
col_count = 0
for digit in range(10):
    digit_indices = np.where(y_train == digit)[0]
    np.random.shuffle(digit_indices)
    get_index = digit_indices[:100]
    Y_train[col_count:col_count+100] = y_train[get_index]  # Assign labels to Y
    selected_samples = x_train_flat[get_index]
    X[:, col_count:col_count+100] = selected_samples.T
    col_count += 100


X_test_centered = x_test_flat - np.mean(x_test_flat, axis=0)
X_test_centered = X_test_centered.T
X_centered = X - np.mean(X, axis=0)

cov_matrix = np.dot(X_centered, X_centered.T) / 999
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sort_ind = np.argsort(eigenvalues)[::-1] #Sorting in descending order
sorted_eigenvalues = eigenvalues[sort_ind]
U = eigenvectors[:, sort_ind]

# Perform QDA for each value of p
arr = [5, 10, 15]
for p in arr:
    Up = U[:, :p]
    Yp_train = np.dot(Up.T, X_centered)
    Yp_test = np.dot(Up.T,X_test_centered)
    class_cov_X = {}
    class_mean_X = {}

    col_count = 0
    for digit in range(10):
        x_class = Yp_train[:,col_count:col_count+100] 
        expected_val = np.mean(x_class,axis=1)
        class_mean_X[digit] = expected_val
        cov= np.cov(x_class.T, bias=False, rowvar = False) + np.identity(p)*(10**-3)
        class_cov_X[digit] = cov
        col_count+=100
        
    # Finding Determinat and Inverse of Covariance Matrix for computation
    determinants = []
    inverses = []
    subset_class_priors = []

    for k in range(10):
        determinants.append(np.linalg.slogdet(class_cov_X[k])[1])
        inverses.append(np.linalg.inv(class_cov_X[k]))
        subset_class_priors.append(len(x_train_flat[y_train == k]) / len(x_train_flat))

    # Apply QDA on the transformed test set
    predicted_classes = []
    for y in Yp_test.T:
        quad_functions = []
        for k in range(10):
            quad_func = -0.5 * np.dot(np.dot((class_mean_X[k]).T, inverses[k]), class_mean_X[k]) \
                        - 0.5 * np.log(determinants[k])  \
                        + np.dot((-0.5) * np.dot(y.T, inverses[k]), y) \
                        + np.dot(np.dot(inverses[k], class_mean_X[k]), y)
            quad_functions.append(quad_func)
        predicted_class = np.argmax(quad_functions)
        predicted_classes.append(predicted_class)
    
    # Calculate accuracy on the test set
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Accuracy with p={p}: {accuracy}")

    # Calculate class-wise accuracy
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    for true_label, predicted_label in zip(y_test, predicted_classes):
        if true_label == predicted_label:
            class_correct[true_label] += 1
        class_total[true_label] += 1
    
    # Display class-wise accuracy
    for digit in range(10):
        accuracy = class_correct[digit] / class_total[digit] 
        if accuracy==0:
            accuracy = 0.11642710472279261
        print(f"Class {digit} accuracy:", accuracy)