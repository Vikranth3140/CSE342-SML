import numpy as np
import csv

mnist_data = np.load('../mnist.npz')

csv_file = 'data.csv'

data_keys = mnist_data.files

with open(csv_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['x_test', 'x_train', 'y_train', 'y_test'])

    for i in range(len(mnist_data['x_test'])):
        x_test = mnist_data['x_test'][i]
        x_train = mnist_data['x_train'][i]
        y_train = mnist_data['y_train'][i]
        y_test = mnist_data['y_test'][i]
        
        csv_writer.writerow([x_test.tolist(), x_train.tolist(), y_train.tolist(), y_test.tolist()])

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