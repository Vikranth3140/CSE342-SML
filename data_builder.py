import numpy as np
import csv

# Load MNIST data
mnist_data = np.load('mnist.npz')

# Specify the CSV file path
csv_file = 'data.csv'

# Extract data keys
data_keys = mnist_data.files

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the column headers
    csv_writer.writerow(['x_test', 'x_train', 'y_train', 'y_test'])

    # Write the data rows
    for i in range(len(mnist_data['x_test'])):
        x_test = mnist_data['x_test'][i]
        x_train = mnist_data['x_train'][i]
        y_train = mnist_data['y_train'][i]
        y_test = mnist_data['y_test'][i]
        
        # Write data row
        csv_writer.writerow([x_test.tolist(), x_train.tolist(), y_train.tolist(), y_test.tolist()])

print("CSV file created successfully.")