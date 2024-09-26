from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt


#Since both datasets have continuous features you will implement decision trees that have binary splits. 
#For determining the optimal threshold for splitting you will need to search over all possible thresholds for
#a given feature (refer to class notes and discussion for an efficient search strategy). Use information gain to measure node impurity in your implementation.


#Growing Decision Trees
#Instead of growing full trees, you will use an early stopping strategy. To this end, wewill impose a limit on the minimum number 
#of instances at a leaf node, let thisthreshold be denoted as nmin, where nmin is described as a percentage relative to thesize 
#of the training dataset. For example, if the size of the training dataset is 150and nmin= 5, then a node will only be split further if it has more than eight instances.
#•For the Iris dataset use nmin E {5, 10, 15, 20}, and calculate the accuracy using10 fold cross-validation for each value of min.
#•For the Spambase dataset use nmin E {5, 10, 15, 20, 25}, and calculate theaccuracy using 10 fold cross-validation for each value of nmin.
#You can summarize your results in two separate tables, one for each dataset (report the average accuracy and standard deviation across the folds).

iris_df = pd.read_csv("iris.csv", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
with pd.option_context('future.no_silent_downcasting', True):
    iris_df= iris_df.replace({'Iris-setosa':0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).infer_objects()

# Decision Tree Function Explanations
## Node Class
class TreeNode():
    def __init__(self, leaf, decision_boundary data, feature, feature_val, prediction_probs, information_gain) -> None:
        #self.data = data
        self.feature = feature
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        #self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None
        self.leaf = False
        self.decision_boundary = decision_boundary

## DecisionTree Class
#This class implements the entire decision tree algorithm.

class DecisionTree():
    """
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features"""

    def __init__(self, min_samples_leaf=1, min_information_gain=0.0)
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.root = None

    def entropy(self, y_1) -> float:
        counts = np.bincount(y_1)
        probabilities = counts / len(y_1)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def information_gain(self, X_column, threshold, y_1):
        parent_entropy = self.entropy(y_1)
        left_mask = X_column <= threshold
        right_mask = X_column > threshold
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        n = len(y)
        n_left, n_right = len(y[left_mask]), len(y[right_mask])
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - child_entropy

    def choose_best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_name in range(X.shape[1]):
            #get all values for each feature 
            X_column = X[:, feature_name]
            #determine all potential threholds
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                #calculate the gain to determine which one will it be the next feature
                gain = self.information_gain(X_column, threshold, y)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_name
                    split_threshold = threshold
        return split_idx, split_threshold   


    def _find_label_probs(self, data: np.array) -> np.array:

        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursive, depth first tree creation algorithm
        """

        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        
        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)
        
        # Find label probs for the node
        label_probabilities = self._find_label_probs(data)

        # Calculate information gain
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        
        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)
        
        return node
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets
        """

        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Start creating the tree
        self.tree = self._create_tree(data=train_data, current_depth=0)

        # Calculate feature importance
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)
        # Normalize the feature importance values
        self.feature_importances = {k: v / total for total in (sum(self.feature_importances.values()),) for k, v in self.feature_importances.items()}

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds    
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        """Calculates the feature importance by visiting each node in the tree recursively"""
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)         

