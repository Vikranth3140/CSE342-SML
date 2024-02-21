import numpy as np
from tqdm import tqdm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def load_mnist_data():
    mnist_data = np.load('../mnist.npz')
    x_train = mnist_data['x_train']
    y_train = mnist_data['y_train']
    x_train = x_train.reshape(-1, 784)
    return x_train, y_train

def load_mnist_test_data():
    mnist_data = np.load('../mnist.npz')
    x_test = mnist_data['x_test']
    y_test = mnist_data['y_test']
    x_test = x_test.reshape(-1, 784)
    return x_test, y_test

def compute_class_statistics(x_train, y_train):
    class_means = []
    class_covariances = []

    class_priors = [len(x_train[y_train==i])/len(x_train) for i in range(10)]

    x_train = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)

    for i in range(10):
        class_samples = x_train[y_train == i]
        class_mean = np.mean(class_samples, axis=0)
        class_covariance = np.cov(class_samples, bias=True, rowvar=False) + np.identity(784) * 1e-10
        class_means.append(class_mean)
        class_covariances.append(class_covariance)

    return class_priors, class_means, class_covariances

def train_qda_sklearn(x_train, y_train):
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(x_train, y_train)
    return qda

def compute_qda_components(priors, mean_vectors, covariance_matrices):
    Wi = []
    wi = []
    wi0 = []

    for prior, mean_vector, cov_matrix in zip(priors, mean_vectors, covariance_matrices):
        cov_matrix_inv = np.linalg.inv(cov_matrix)

        Wi_i = -0.5 * cov_matrix_inv
        wi_i = np.dot(cov_matrix_inv, mean_vector)
        wi0_i = -0.5 * np.dot(np.dot(mean_vector.T, cov_matrix_inv), mean_vector) - 0.5 * np.linalg.slogdet(cov_matrix)[1] + np.log(prior)

        Wi.append(Wi_i)
        wi.append(wi_i)
        wi0.append(wi0_i)

    return Wi, wi, wi0

def qda_classify(x, Wi, wi, wi0):
    x = x.reshape(-1, 1)
    scores = []
    for i in range(len(Wi)):
        score = np.dot(np.dot(x.T, Wi[i]), x) + np.dot(wi[i].T, x) + wi0[i]
        scores.append(score)
    return np.argmax(scores)

x_train, y_train = load_mnist_data()
print(f"x_train size: {x_train.size}")
print(f"x_train shape: {x_train.shape}")

x_test, y_test = load_mnist_test_data()
print(f"x_test size: {x_test.size}")
print(f"x_test shape: {x_test.shape}")

### custom impl ###
priors, means, covs = compute_class_statistics(x_train, y_train)
Wi, wi, wi0 = compute_qda_components(priors, means, covs)
preds = []
classwise_accs = {i:[] for i in range(10)}
for i in tqdm(range(x_test.shape[0])):
    predicted_class = qda_classify(x_test[i], Wi, wi, wi0)
    preds.append(predicted_class)
    classwise_accs[y_test[i]].append(predicted_class)

y_pred_custom = np.asarray(preds)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"\nClass Implementation Accuracy: {accuracy_custom}")
for i in range(10):
    y = np.asarray(classwise_accs[i])
    print('Class {} accuracy:'.format(i))
    print(y[y==i].shape[0]/len(y))

### sklearn impl ###
qda = train_qda_sklearn(x_train, y_train)
y_pred_sklearn = qda.predict(x_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"\Sklean Implementation Accuracy: {accuracy_sklearn}")