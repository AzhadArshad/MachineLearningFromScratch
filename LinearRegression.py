import numpy as np

class LinearRegression:
    def __init__(self, lr=0.1,epochs=1000):
        self.lr = lr # Learning rate
        self.epochs = epochs # Number of epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        

    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
    #mse = np.mean((y_pred - y) ** 2)