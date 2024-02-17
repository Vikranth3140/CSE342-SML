import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def load_mnist_data():
    mnist_data = np.load('../mnist.npz')
    x_train = mnist_data['x_train']
    y_train = mnist_data['y_train']
    x_train = x_train.reshape((x_train.shape[0], -1))
    return x_train, y_train

def compute_class_statistics(x_train, y_train):
    class_means = []
    class_covariances = []

    for i in range(10):
        class_samples = x_train[y_train == i]
        class_mean = np.mean(class_samples, axis=0)
        class_covariance = np.cov(class_samples.T)
        class_means.append(class_mean)
        class_covariances.append(class_covariance)

    return class_means, class_covariances

def print_class_statistics(class_means, class_covariances):
    print("Mean vectors:")
    print()
    for i, mean_vec in enumerate(class_means):
        print()
        print(f"Class {i}")
        print()
        print(mean_vec)

    print()
    print("\nCovariance matrices:")
    for i, cov_mat in enumerate(class_covariances):
        print()
        print(f"Class {i}:")
        print()
        print(cov_mat)

def train_qda(x_train, y_train):
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(x_train, y_train)
    return qda

x_train, y_train = load_mnist_data()
class_means, class_covariances = compute_class_statistics(x_train, y_train)
print()
print_class_statistics(class_means, class_covariances)
qda = train_qda(x_train, y_train)