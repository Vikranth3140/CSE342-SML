import numpy as np

def load_mnist_data():
    mnist_data = np.load('../mnist.npz')
    x_train = mnist_data['x_train']
    y_train = mnist_data['y_train']
    x_train = x_train.reshape((x_train.shape[0], -1))
    return x_train, y_train

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

    print("Shape of X:", X.shape)
    print("Shape of X_recon:", X_recon.shape)
    print("MSE between X and X_recon:", mse)