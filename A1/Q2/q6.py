import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

def load_mnist_data():
    mnist_data = np.load('../mnist.npz')
    x_train = mnist_data['x_train']
    y_train = mnist_data['y_train']
    x_train = x_train.reshape((x_train.shape[0], -1))
    return x_train, y_train

def load_mnist_test_data():
    mnist_data = np.load('../mnist.npz')
    x_test = mnist_data['x_test']
    y_test = mnist_data['y_test']
    x_test = x_test.reshape((x_test.shape[0], -1))
    return x_test, y_test

def create_data_matrix(x_train, y_train):
    samples = []
    for i in range(10):
        class_samples = x_train[y_train == i][:100]
        samples.append(class_samples)
    Y = np.concatenate(samples, axis=0)
    X = Y.T
    return X, len(Y)

if __name__ == "__main__":
    x_train, y_train = load_mnist_data()
    X, num_samples = create_data_matrix(x_train, y_train)

    mean_X = np.mean(X, axis=0)
    X = X - mean_X

    S = np.dot(X, X.T) / 999

    eigenvalues, eigenvectors = np.linalg.eig(S)

    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    U = eigenvectors

    Y = np.dot(U.T, X)
    U = U[:, :num_samples]

    X_recon = np.dot(U, Y)
    mse = np.mean((X - X_recon) ** 2)

    p_values = [5, 10, 20]

    for p in p_values:
        U_p = U[:, :p]

        Y_p = np.dot(U_p.T, X)

        X_recon_p = np.dot(U_p, Y_p)

        X_recon_p = X_recon_p + mean_X

        X_recon_p = np.real(X_recon_p.astype(np.float64))
        X_recon_p = X_recon_p.reshape((-1, 28, 28))

        fig, axs = plt.subplots(10, 5, figsize=(10, 20))

        for i in range(10):
            samples = X_recon_p[y_train[:1000] == i][:5]
            
            for j in range(5):
                axs[i, j].imshow(samples[j], cmap='gray')
                axs[i, j].axis('off')

        plt.suptitle(f"Reconstructed images with p={p}")
        plt.show()


    x_test, y_test = load_mnist_test_data()

    Y_test = np.dot(U_p.T, x_test)

    p_values = [5, 10, 20]

    for p in p_values:
            U_p = U[:, :p]

            Y_test_p = np.dot(U_p.T, x_test.T)

            Y_test_p = Y_test_p.T

            # Apply QDA on transformed test data
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(Y.T, y_train)
            y_pred = qda.predict(Y_test_p)

            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy with {p} principal components: {test_accuracy}")

            cm = confusion_matrix(y_test, y_pred)
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
            print(f"Per-class accuracy with {p} principal components:")
            for i in range(10):
                print(f"Class {i}: {per_class_accuracy[i]}")