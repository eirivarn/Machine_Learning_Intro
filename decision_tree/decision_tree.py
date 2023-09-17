import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
from collections import Counter

class Node:
    def __init__(self, feature=None, is_leaf=False, class_label=None):
        """
        Initialize a node for the decision tree.
        """
        self.feature = feature  
        self.children = {}  
        self.is_leaf = is_leaf  
        self.class_label = class_label  #
        

class DecisionTree:
    
    def __init__(self):
        """Initialize the DecisionTree_Original instance."""
        self.root = None  
        self.feature_importances_ = dict()

    def build_tree(self, X, y, features):
        """
        Recursively construct the decision tree by choosing the feature that 
        maximizes the entropy reduction at each step.
        
        Args:
        - X (pd.DataFrame): Features.
        - y (pd.Series): Labels.
        - features (list): List of features to consider for splitting.
        
        Returns:
        - Node: The constructed decision tree node.
        """
        best_gain = -1
        best_feature = None

        # Base case: Return a leaf node if all labels are pure or no features left to split
        unique_labels = y.unique()
        if len(unique_labels) == 1 or not features:
            return Node(is_leaf=True, class_label=y.iloc[0])
        
        # Find the best feature based on entropy reduction
        for feature in features:
            gain = entropy_reduction(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        # If no gain, return a leaf node with the most common label
        if best_gain == 0:
            most_common_label = y.mode()[0]
            return Node(is_leaf=True, class_label=most_common_label)

        # Else split on best feature
        node = Node(feature=best_feature)

        if best_feature in self.feature_importances_:
            self.feature_importances_[best_feature] += best_gain
        else:
            self.feature_importances_[best_feature] = best_gain

        # Recursively build tree for each unique value of the best feature
        for value in X[best_feature].unique():
            subset_data = X[X[best_feature] == value]
            subset_target = y[subset_data.index]
            
            remaining_features = []
            for feature in features:
                if feature != best_feature:
                    remaining_features.append(feature)
            
            child_node = self.build_tree(subset_data, subset_target, remaining_features)
            node.children[value] = child_node

        return node

    def fit(self, X, y):
        """
        Build the decision tree using the training data.
        
        Args:
        - X (pd.DataFrame): Features.
        - y (pd.Series): Labels.
        """
        features = X.columns.tolist()
        self.root = self.build_tree(X, y, features)

    def predict_sample(self, row, node):
        """
        Predict label for a single data point.
        
        Args:
        - row (pd.Series): Single data point.
        - node (Node): Current decision tree node.
        
        Returns:
        - Label for the data point.
        """
        if node.is_leaf:
            return node.class_label
        
        feature_value = row[node.feature]
        
        # Recur down the tree if current feature value matches a child node
        if feature_value in node.children:
            return self.predict_sample(row, node.children[feature_value])

    def predict(self, X):
        """
        Predict labels for a dataset.
        
        Args:
        - X (pd.DataFrame): Input data.
        
        Returns:
        - pd.Series: Predicted labels.
        """
        predictions = []
        for index, row in X.iterrows():
            predicted_label = self.predict_sample(row, self.root)
            predictions.append(predicted_label)
            
        return pd.Series(predictions, index=X.index)

    def get_rules(self, node=None, current_path=[]):
        """
        Extract rules from the decision tree.
        
        Args:
        - node (Node, optional): Current node. Defaults to root.
        - path (list, optional): Current path of features and values.
        
        Returns:
        - list: List of rules.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return [(current_path, node.class_label)]
        
        # Initialize a list to store the rules
        rules = [] # = decision rules

        for value, child_node in node.children.items():
            new_path = current_path + [(node.feature, value)]
            rules.extend(self.get_rules(child_node, new_path))
        
        return rules

    def get_feature_importance(self):
        """Returns the features sorted by their importance."""
        return dict(sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True))


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()

def entropy(y):
    """
    Computes the entropy of a partitioning

    Args:
        y (pd.Series): Series of class labels

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    """
    counts = y.value_counts()
    probs = counts / len(y)
    return - np.sum(probs * np.log2(probs))

def entropy_reduction(X, y, feature):
        """
        Calculate the entropy reduction when splitting on a given feature.
        
        Args:
        - X (pd.DataFrame): Features.
        - y (pd.Series): Labels.
        - feature (str): The feature to compute entropy reduction for.
        
        Returns:
        - float: The reduction in entropy.
        """
        n_values = np.unique(X[feature])
        subsets_entropy_sum = 0

        # Compute entropy for each unique value of the feature
        for value in n_values:
            subset = y[X[feature] == value]
            weight = len(subset) / len(y)
            subsets_entropy_sum += weight * entropy(subset)

        return entropy(y) - subsets_entropy_sum

