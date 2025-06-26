import numpy as np
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def relu(Z):
    # ReLU activation: sets negative values to 0
    return np.maximum(0, Z)

def relu_derivative(Z):
    # Derivative of ReLU: 1 for positive Z, 0 for negative
    return Z > 0

def sigmoid(Z):
    # Sigmoid activation: maps values to range (0, 1)
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    # Derivative of sigmoid: used for backprop in output layer
    s = sigmoid(Z)
    return s * (1 - s)

# Neural Network class definition
class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        # Set learning rate
        self.lr = learning_rate

        # Initialize weights with small random values and biases with zeros
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros((1, hidden1_size))

        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros((1, hidden2_size))

        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward propagation through the network

        # Layer 1: Input → Hidden1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)

        # Layer 2: Hidden1 → Hidden2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)

        # Layer 3: Hidden2 → Output
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)  # Final output probability (for binary classification)

        return self.A3

    def compute_loss(self, Y, A3):
        # Binary cross-entropy loss function
        m = Y.shape[0]  # number of examples
        return -1/m * np.sum(Y * np.log(A3 + 1e-8) + (1 - Y) * np.log(1 - A3 + 1e-8))

    def backward(self, X, Y):
        # Backward propagation to compute gradients

        m = X.shape[0]  # number of examples

        # Output layer
        dZ3 = self.A3 - Y
        dW3 = (1/m) * np.dot(self.A2.T, dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)

        # Hidden layer 2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, Y, epochs=1000):
        # Train the neural network

        self.train_losses = []

        for i in range(epochs):
            # Forward pass
            A3 = self.forward(X)
            # Compute training loss
            train_loss = self.compute_loss(Y, A3)
            self.train_losses.append(train_loss)
   

            # Backward pass (update weights)
            self.backward(X, Y)

            # Print loss every 100 epochs
            if i % 100 == 0:
                print(f"Epoch {i}, Train Loss: {train_loss:.4f}", end="")
                print()

    def predict(self, X):
        # Predict class labels (0 or 1) using forward pass
        probs = self.forward(X)
        return (probs > 0.5).astype(int)  # Threshold at 0.5

    def plot_learning_curve(self):
        # Plot training and validation loss curves

        if not hasattr(self, 'train_losses'):
            print("Train the model first to generate learning curves.")
            return

        plt.plot(self.train_losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.show()





    

