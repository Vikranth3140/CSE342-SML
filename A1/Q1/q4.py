import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

def load_mnist_data():
    mnist_data = np.load('../mnist.npz')
    x_train = mnist_data['x_train'].reshape((len(mnist_data['x_train']), -1))
    y_train = mnist_data['y_train']
    x_test = mnist_data['x_test'].reshape((len(mnist_data['x_test']), -1))
    y_test = mnist_data['y_test']
    return x_train, y_train, x_test, y_test

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_wise_accuracy = np.diag(cm) / cm.sum(axis=1)
    total_samples_tested = len(y_test)
    return accuracy, class_wise_accuracy, total_samples_tested

def print_results(accuracy, class_wise_accuracy, total_samples_tested):
    print(f"Overall accuracy: {accuracy}")
    print()
    for i, acc in enumerate(class_wise_accuracy):
        print(f"Accuracy for class {i}: {acc}")
    print()
    print(f"Total number of samples tested: {total_samples_tested}")

def main():
    x_train, y_train, x_test, y_test = load_mnist_data()
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(x_train, y_train)
    accuracy, class_wise_accuracy, total_samples_tested = evaluate_model(qda, x_test, y_test)
    print()
    print_results(accuracy, class_wise_accuracy, total_samples_tested)

if __name__ == "__main__":
    main()