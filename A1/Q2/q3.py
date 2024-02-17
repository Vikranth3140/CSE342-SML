import numpy as np

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

if __name__ == "__main__":
    file_path = '../mnist.npz'
    x_train, y_train = load_mnist_data(file_path)
    X, num_samples = create_data_matrix(x_train, y_train)
    
    mean_X = np.mean(X, axis=0)
    X = X - mean_X
    
    S = np.cov(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(S)

    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    U = eigenvectors

    print()
    print("Matrix U created with shape:", U.shape)
    print()
    print(U)