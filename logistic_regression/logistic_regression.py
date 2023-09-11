import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, learning_rate=10, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent with Learning Rate Decay
        for _ in range(self.n_iterations):
            # Predict current outputs
            model_output = sigmoid(np.dot(X, self.weights) + self.bias)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (model_output - y))
            db = (1 / n_samples) * np.sum(model_output - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_output)

        
class LogisticRegressionWithPolyFeatures(LogisticRegression):
    def __init__(self, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
    
    def fit(self, X, y):
        # Generate polynomial features
        X_poly = polynomial_features(X, self.degree)
        super().fit(X_poly, y)
    
    def predict(self, X):
        # Generate polynomial features
        X_poly = polynomial_features(X, self.degree)
        return super().predict(X_poly)
        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float) 
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))


def combinations_with_replacement(iterable, r):
    """
    Return r-length combinations of elements from the iterable allowing individual elements to be repeated more than once.
    
    Args:
        iterable: Input iterable data.
        r (int): Length of the combinations.
        
    Returns:
        Combinations with replacement.
    """
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)


def polynomial_features(X, degree=2):
    n_samples, n_features = X.shape
    new_features = [X]
    
    for d in range(2, degree + 1):
        for feature_indices in combinations_with_replacement(range(n_features), d):
            new_feature = np.prod(X[:, feature_indices], axis=1)
            new_features.append(new_feature.reshape(n_samples, 1))
    
    return np.hstack(new_features)