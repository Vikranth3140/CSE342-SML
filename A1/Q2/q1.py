import numpy as np

mnist_data = np.load('../mnist.npz')

x_train = mnist_data['x_train']
y_train = mnist_data['y_train']

x_train = x_train.reshape((x_train.shape[0], -1))

samples = []

for i in range(10):
    class_samples = x_train[y_train == i][:100]

    samples.append(class_samples)

Y = np.concatenate(samples, axis=0)

X = Y.T

print("Data matrix X created with shape:", X.shape)
print("Number of samples:", len(Y))
print(X)