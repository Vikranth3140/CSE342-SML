import numpy as np
from sklearn.metrics import mean_squared_error

def load_mnist_data(file_path):
    mnist_data = np.load(file_path)
    x_train = mnist_data['x_train']
    y_train = mnist_data['y_train']
    x_train = x_train.reshape((x_train.shape[0], -1))
    return x_train, y_train

def create_data_matrix(x_train, y_train, num_samples_per_class=100):
    samples = []
    for i in range(10):
        class_samples = x_train[y_train == i][:num_samples_per_class]
        samples.append(class_samples)
    Y = np.concatenate(samples, axis=0)
    X = Y.T
    return X, len(Y)

def perform_pca(X, U):
    Y = np.dot(U.T, X.T)
    X_recon = np.dot(U, Y)
    mse = mean_squared_error(X, X_recon.T)
    return Y, mse

if __name__ == "__main__":
    file_path = '../mnist.npz'
    x_train, y_train = load_mnist_data(file_path)
    X, num_samples = create_data_matrix(x_train, y_train)

    X = X.astype(float)

    X -= np.mean(X, axis=0)

    S = np.dot(X, X.T) / (num_samples - 1)

    eigenvalues, eigenvectors = np.linalg.eig(S)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    U = eigenvectors

    Y = np.dot(U.T, X)

    X_recon = np.dot(U, Y)

    mse = np.mean((X - X_recon) ** 2)
    print("MSE between X and X_recon:", mse)