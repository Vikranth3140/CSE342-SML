import numpy as np
import csv
from numpy import load

data = load('mnist.npz')
lst = data.files

# Specify the CSV file path
csv_file = 'data.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the column headers
    writer.writerow(['x_test', 'x_train', 'y_train', 'y_test'])

    # Write the data rows
    for i in range(len(data['x_test'])):
        writer.writerow([data['x_test'][i], data['x_train'][i], data['y_train'][i], data['y_test'][i]])