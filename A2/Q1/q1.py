import numpy as np

try:
    with np.load('../mnist.npz') as mnist_data:
        test_samples_per_class = np.bincount(mnist_data['y_test'])
        train_samples_per_class = np.bincount(mnist_data['y_train'])

    print()
    print("Total number of test samples from each class:")
    for class_label, count in enumerate(test_samples_per_class):
        print(f"Class {class_label}: {count} samples")

    print()
    print("Total number of train samples from each class:")
    for class_label, count in enumerate(train_samples_per_class):
        print(f"Class {class_label}: {count} samples")

    print()
    total_test_samples = np.sum(test_samples_per_class)
    total_train_samples = np.sum(train_samples_per_class)

    print(f"Total number of test samples: {total_test_samples}")
    print(f"Total number of train samples: {total_train_samples}")

except FileNotFoundError:
    print("Error: mnist.npz file not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")