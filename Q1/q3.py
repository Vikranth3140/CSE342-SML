import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# Load MNIST data
mnist_data = np.load('../mnist.npz')

# Extract data
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']
x_test = mnist_data['x_test']
y_test = mnist_data['y_test']

# Vectorize the images
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Create a QDA object
qda = QuadraticDiscriminantAnalysis(store_covariance=True)

# Fit the QDA model
qda.fit(x_train, y_train)

# Predict the classes of the test samples
y_pred = qda.predict(x_test)

# Compute the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {accuracy}")

# Compute the class-wise accuracy
cm = confusion_matrix(y_test, y_pred)
class_wise_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(class_wise_accuracy):
    print(f"Accuracy for class {i}: {acc}")