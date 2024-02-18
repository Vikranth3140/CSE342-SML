import numpy as np

def load_mnist_data():
    with np.load('../mnist.npz') as mnist_data:
        test_samples_per_class = count_samples_per_class(mnist_data['y_test'])
        train_samples_per_class = count_samples_per_class(mnist_data['y_train'])
    return test_samples_per_class, train_samples_per_class

def count_samples_per_class(samples):
    samples_per_class = np.bincount(samples)
    return samples_per_class

def print_samples_per_class(samples_per_class, dataset_name):
    print()
    print(f"Total number of {dataset_name} samples from each class:")
    for class_label, count in enumerate(samples_per_class):
        print(f"Class {class_label}: {count} samples")

def calculate_total_samples(samples_per_class):
    print()
    total_samples = np.sum(samples_per_class)
    return total_samples

test_samples_per_class, train_samples_per_class = load_mnist_data()

print_samples_per_class(test_samples_per_class, "test")
print_samples_per_class(train_samples_per_class, "train")

total_test_samples = calculate_total_samples(test_samples_per_class)
total_train_samples = calculate_total_samples(train_samples_per_class)

print(f"Total number of test samples: {total_test_samples}")
print(f"Total number of train samples: {total_train_samples}")