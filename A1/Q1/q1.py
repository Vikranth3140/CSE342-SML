import numpy as np

mnist_data = np.load('../mnist.npz')

test_samples_per_class = np.bincount(mnist_data['y_test'])
train_samples_per_class = np.bincount(mnist_data['y_train'])

print("Total number of test samples from each class:")
for class_label, count in enumerate(test_samples_per_class):
    print(f"Class {class_label}: {count} samples")

print("Total number of train samples from each class:")
for class_label, count in enumerate(train_samples_per_class):
    print(f"Class {class_label}: {count} samples")

total_test_samples = np.sum(test_samples_per_class)
total_train_samples = np.sum(train_samples_per_class)

print(f"Total number of test samples: {total_test_samples}")
print(f"Total number of train samples: {total_train_samples}")