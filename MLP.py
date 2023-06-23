import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the output labels to one-hot encoded vectors
y_train_onehot = np.zeros((y_train.size, y_train.max()+1))
y_train_onehot[np.arange(y_train.size), y_train] = 1
y_test_onehot = np.zeros((y_test.size, y_test.max()+1))
y_test_onehot[np.arange(y_test.size), y_test] = 1

# Define the MLP structure
input_size = X_train.shape[1]
hidden_size = 5
output_size = y_train_onehot.shape[1]
lr = 0.01
epochs = 50

# Initialize the weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Train the MLP using Backpropagation and Gradient Descent
train_accs, test_accs = [], []
train_precisions, test_precisions = [], []

for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Backward propagation
    error = y_train_onehot - a2
    delta2 = error * a2 * (1 - a2)
    delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)

    # Update the weights and biases
    W2 += lr * np.dot(a1.T, delta2)
    b2 += lr * np.sum(delta2, axis=0, keepdims=True)
    W1 += lr * np.dot(X_train.T, delta1)
    b1 += lr * np.sum(delta1, axis=0)

    # Evaluate the training set
    z1_train = np.dot(X_train, W1) + b1
    a1_train = sigmoid(z1_train)
    z2_train = np.dot(a1_train, W2) + b2
    a2_train = sigmoid(z2_train)
    predictions_train = np.argmax(a2_train, axis=1)
    accuracy_train = np.mean(predictions_train == y_train)
  
    # Evaluate the testing set
    z1_test = np.dot(X_test, W1) + b1
    a1_test = sigmoid(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    a2_test = sigmoid(z2_test)
    predictions_test = np.argmax(a2_test, axis=1)
    accuracy_test = np.mean(predictions_test == y_test)
   
    # Append the evaluation 
    train_accs.append(accuracy_train)
    test_accs.append(accuracy_test)
  

# Plot the evaluation metrics
epochs_list = range(1, epochs+5)

plt.plot(epochs_list, train_accs, label='Train')
plt.plot(epochs_list, test_accs, label='Test')

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

