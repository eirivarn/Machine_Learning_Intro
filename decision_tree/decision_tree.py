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

#Original DecisionTree

class DecisionTree_Original:
    
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

        # Ensure that node has valid children
        if not node.children:
            return Node(is_leaf=True, class_label=y.mode()[0])

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
        
        # If the current feature value matches a child node, proceed with that child node
        if feature_value in node.children:
            return self.predict_sample(row, node.children[feature_value])

        # If no match is found, return the most common label from the leaf children of the current node
        leaf_labels = [child.class_label for child in node.children.values() if child.is_leaf]

        # If there are no leaf_labels, default to the most common label from the entire dataset (although this shouldn't occur)
        if not leaf_labels:
            most_common_label = None  # Or you can assign the most common label from your entire dataset
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
        return pd.Series(predictions, index=X.index)


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


#DecisionTree with minimumsplit and max depth
class DecisionTree_Minsplit_Maxdepth(DecisionTree_Original):
    
    def __init__(self, min_samples_split=2, max_depth=None):
        super().__init__()  # Call the base class constructor
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    
    def build_tree(self, X, y, features, current_depth=0):
        best_gain = -1
        best_feature = None

        # Check if tree has reached max depth or there are not enough samples for a split
        if (self.max_depth and current_depth >= self.max_depth) or (len(y) < self.min_samples_split):
            return Node(is_leaf=True, class_label=y.mode()[0])

        # If all samples have the same label or no features left to consider
        if len(y.unique()) == 1 or not features:
            return Node(is_leaf=True, class_label=y.iloc[0])
        
        # Find the feature with the highest information gain.
        for feature in features:
            gain = entropy_reduction(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        # If information gain is 0, return a leaf node with the most common label
        if best_gain == 0:
            return Node(is_leaf=True, class_label=y.mode()[0])

        node = Node(feature=best_feature)

        for value in X[best_feature].unique():
            subset_data = X[X[best_feature] == value]
            subset_target = y[subset_data.index]
            remaining_features = list(set(features) - {best_feature})
            child_node = self.build_tree(subset_data, subset_target, remaining_features, current_depth+1)
            node.children[value] = child_node

        #ensure that node is properly initialized
        if node is None:
            return Node(is_leaf=True, class_label=y.mode()[0])
        
        return node


class DecisionTree_Prune(DecisionTree_Original):

    def __init__(self, min_samples_split=2, max_depth=None):
        super().__init__()  # Call the base class constructor
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def prune(self, node, X_valid, y_valid):
        """
        Post-pruning of the decision tree using reduced error pruning.
        """
        if not node.is_leaf:
            # Create a copy of the current node to test its pruning
            pruned_node = Node(is_leaf=True, class_label=y_valid.mode()[0])
            
            original_predictions = [self.predict_sample(row, node) for _, row in X_valid.iterrows()]
            pruned_predictions = [self.predict_sample(row, pruned_node) for _, row in X_valid.iterrows()]
            
            original_accuracy = accuracy(y_valid, pd.Series(original_predictions, index=y_valid.index))
            pruned_accuracy = accuracy(y_valid, pd.Series(pruned_predictions, index=y_valid.index))
            
            # If pruning this node improves or maintains accuracy, prune it
            if pruned_accuracy >= original_accuracy:
                node.is_leaf = True
                node.class_label = y_valid.mode()[0]
                node.children = {}
            else:
                for value, child_node in node.children.items():
                    subset_valid = X_valid[X_valid[node.feature] == value]
                    subset_valid_target = y_valid[subset_valid.index]
                    self.prune(child_node, subset_valid, subset_valid_target)
    
    def fit(self, X, y, X_valid=None, y_valid=None):
        """
        Generates a decision tree for classification and prunes it using a validation set.
        """
        super().fit(X, y)  # Call the fit method from the parent class
        if X_valid is not None and y_valid is not None:
            self.prune(self.root, X_valid, y_valid)


class DecisionTree_Combined(DecisionTree_Original):

    def __init__(self, min_samples_split=2, max_depth=None):
        super().__init__()  # Call the base class constructor
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, X, y, features, current_depth=0):
        best_gain = -1
        best_feature = None

        if (self.max_depth and current_depth >= self.max_depth) or (len(y) < self.min_samples_split):
            return Node(is_leaf=True, class_label=y.mode()[0])

        if len(y.unique()) == 1 or not features:
            return Node(is_leaf=True, class_label=y.iloc[0])

        for feature in features:
            gain = entropy_reduction(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_gain == 0:
            return Node(is_leaf=True, class_label=y.mode()[0])

        node = Node(feature=best_feature)

        for value in X[best_feature].unique():
            subset_data = X[X[best_feature] == value]
            subset_target = y[subset_data.index]
            remaining_features = list(set(features) - {best_feature})
            child_node = self.build_tree(subset_data, subset_target, remaining_features, current_depth+1)
            node.children[value] = child_node

        return node

    def prune(self, node, X_valid, y_valid):
        if not node.is_leaf:
            for value in X_valid[node.feature].unique():  # Iterate over unique values
                subset_valid = X_valid[X_valid[node.feature] == value]

                subset_valid_target = y_valid.head(len(subset_valid))
                y_valid = y_valid.iloc[len(subset_valid):]  # remove the rows we've just used
                
                pruned_node = Node(is_leaf=True, class_label=subset_valid_target.mode()[0])

                original_predictions = [self.predict_sample(row, node) for _, row in subset_valid.iterrows()]
                pruned_predictions = [self.predict_sample(row, pruned_node) for _, row in subset_valid.iterrows()]

                original_accuracy = accuracy(subset_valid_target, pd.Series(original_predictions, index=subset_valid_target.index))
                pruned_accuracy = accuracy(subset_valid_target, pd.Series(pruned_predictions, index=subset_valid_target.index))

                if pruned_accuracy >= original_accuracy:
                    node.is_leaf = True
                    node.class_label = subset_valid_target.mode()[0]
                    node.children = {}
                else:
                    child_node = node.children.get(value)
                    if child_node:
                        self.prune(child_node, subset_valid, subset_valid_target)

    def fit(self, X, y, X_valid=None, y_valid=None):
        """
        Generates a decision tree for classification and prunes it using a validation set.
        """
        super().fit(X, y)  # Call the fit method from the parent class
        if X_valid is not None and y_valid is not None:
            self.prune(self.root, X_valid, y_valid)
