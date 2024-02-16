import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

mnist_data = np.load('../mnist.npz')

x_train = mnist_data['x_train']
y_train = mnist_data['y_train']

x_train = x_train.reshape((x_train.shape[0], 784))

class_means = []
class_covariances = []

for i in range(10):
    class_samples = x_train[y_train == i]
    class_mean = np.mean(class_samples, axis=0)
    class_covariance = np.cov(class_samples.T)
    class_means.append(class_mean)
    class_covariances.append(class_covariance)

print("Mean vectors:")
for i, mean_vec in enumerate(class_means):
    print(f"Class {i}: {mean_vec}")

print("\nCovariance matrices:")
for i, cov_mat in enumerate(class_covariances):
    print(f"Class {i}: {cov_mat}")

qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(x_train, y_train)