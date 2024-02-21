import numpy as np
import matplotlib.pyplot as plt

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

    mean_X = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_X

    S = np.dot(X_centered, X_centered.T) / (num_samples - 1)

    eigenvalues, eigenvectors = np.linalg.eig(S)

    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    p_values = [5, 10, 20]

    for p in p_values:
        U_p = eigenvectors[:, :p]

        Y_p = np.dot(U_p.T, X_centered)

        X_recon_p = np.dot(U_p, Y_p) + mean_X
        X_recon_p = np.real(X_recon_p).astype(np.float64)

        images_per_class = 5
        fig, axs = plt.subplots(10, images_per_class, figsize=(10, 20))
        for i in range(10):
            class_indices = np.where(y_train[:1000] == i)[0][:images_per_class]
            for j, idx in enumerate(class_indices):
                image = X_recon_p[:, idx].reshape(28, 28)
                axs[i, j].imshow(image, cmap='gray')
                axs[i, j].axis('off')

        plt.suptitle(f"Reconstructed images with p={p}")
        plt.show()