import numpy as np
import matplotlib.pyplot as plt

NUM_SAMPLES = 5
NUM_CLASSES = 10

def load_mnist_data():
    with np.load('../mnist.npz') as mnist_data:
        x_train = mnist_data['x_train']
        y_train = mnist_data['y_train']
    return x_train, y_train

def plot_samples(x_train, y_train):
    fig, axs = plt.subplots(NUM_CLASSES, NUM_SAMPLES, figsize=(10, 20))

    for i in range(NUM_CLASSES):
        samples = x_train[y_train == i][:NUM_SAMPLES]

        for j in range(NUM_SAMPLES):
            axs[i, j].imshow(samples[j], cmap='gray')
            axs[i, j].axis('off')

    plt.show()

if __name__ == '__main__':
    x_train, y_train = load_mnist_data()
    plot_samples(x_train, y_train)