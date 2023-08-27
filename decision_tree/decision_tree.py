import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
from collections import Counter

class Node:
    def __init__(self, feature=None, is_leaf=False, class_label=None):
        self.feature = feature
        self.children = {}
        self.is_leaf = is_leaf
        self.class_label = class_label

class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.root = None
        pass
    

    def build_tree(self, X, y, features):
        best_gain = -1
        best_feature = None

        # If all samples have the same label or no features left to consider
        if len(y.unique()) == 1 or not features:
            return Node(is_leaf=True, class_label=y.iloc[0])
        
        # Find the feature with the highest information gain.
        for feature in features:
            gain = entropy_reduction(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        print(best_feature)

        # If information gain is 0, return a leaf node with the most common label
        if best_gain == 0:
            return Node(is_leaf=True, class_label=y.mode()[0])

        #setting up the structure to hold the best feature to split on and its corresponding branches
        node = Node(feature=best_feature)

        # For every distinct value in the best feature, partition the data and grow our tree.
        for value in X[best_feature].unique():

            # Create a subset of the data where the best feature equals the current value.
            subset_data = X[X[best_feature] == value]
            subset_target = y[subset_data.index]

            # Convert the list of features into a set, then subtract the set containing the best_feature, and finally convert it back to a list.
            remaining_features = list(set(features) - {best_feature})

            # Recursive call to build the subtree for this branch.
            child_node = self.build_tree(subset_data, subset_target, remaining_features)
            node.children[value] = child_node

        return node

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        features = X.columns.tolist()
        self.root = self.build_tree(X, y, features)
    
    def predict_sample(self, row, node):
        if node.is_leaf:
            return node.class_label
        
        feature_value = row[node.feature]
        if feature_value in node.children:
            return self.predict_sample(row, node.children[feature_value])
        
        # This line is a list comprehension, a compact way to construct a list. 
        # It iterates over each child node of the current node.
        # If the child is a leaf it takes its class label and adds it to the leaf_labels list.
        leaf_labels = [child.class_label for child in node.children.values() if child.is_leaf]

        # If there are no leaf_labels, set most_common_label to None
        if not leaf_labels:
            most_common_label = None
        else:
            # Use a Counter to find the most common label among leaf nodes
            label_counts = Counter(leaf_labels)
            most_common_label = label_counts.most_common(1)[0][0]

        return most_common_label
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        predictions = []
        for index, row in X.iterrows():
            predictions.append(self.predict_sample(row, self.root))
        return pd.Series(predictions)


    def get_rules(self, node=None, path=[]):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return [(path, node.class_label)]

        rules = []
        for value, child_node in node.children.items():
            # Append the current decision to the path
            extended_path = path + [(node.feature, value)]
            rules.extend(self.get_rules(child_node, extended_path))

        return rules



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


def entropy_reduction(X, y, features):
        #Get all the unique values of a feature
        n_values = np.unique(X[features])
        subsets_entropy_sum = 0

        #Sum the entropy of the subsets based on unique values
        for value in n_values:
            subset = y[X[features] == value]
            weight = len(subset) / len(y)
            subsets_entropy_sum += weight*entropy(subset)

        return entropy(y) - subsets_entropy_sum
