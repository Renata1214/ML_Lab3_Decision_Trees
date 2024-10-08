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
X_iris = iris_df.drop(columns=["class"]).values  # All columns except "class"
y_iris = iris_df["class"].values 

# Decision Tree Function Explanations
## Node Class
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self. predict_val= None

## DecisionTree Class
#This class implements the entire decision tree algorithm.

class DecisionTree():
    """
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features"""

    def __init__(self, min_samples_split, min_information_gain=0.0)
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.root = None

    def entropy(self, y_1) -> float:
        counts = np.bincount(y_1)
        probabilities = counts / len(y_1)
        total_sum = 0
        for p in probabilities:
            if p > 0:
                total_sum += p * np.log2(p)
        result = -total_sum
        return result

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

    def best_split(self, X, y):
        best = -1
        split_id, split_threshold = None, None
        for feature_name in range(X.shape[1]):
            #get all values for each feature 
            X_column = X[:, feature_name]
            #determine all potential threholds
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                #calculate the gain to determine which one will it be the next feature
                gain = self.information_gain(X_column, threshold, y)
                if gain > best:
                    best = gain
                    split_id = feature_name
                    split_threshold = threshold
        return split_id, split_threshold  

    def most_occuring_label(self, y):
        label_counts = {}
        most_frequent_label = None
        max_count = -1
        # Count the occurrences of each label in y
        for label in y:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        # Find the label with the highest count
        for label1, count in label_counts.items():
            if count > max_count:
                max_count = count
                most_frequent_label = label1
        return most_frequent_label 

    def build_tree(self, X, y):
        if len(np.unique(y)) == 1 or X.shape[0] < self.min_samples_split:
            obj1=TreeNode(value=self.most_occuring_label(y))
            obj1.leaf = True
            return obj1
        best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None:
            return TreeNode(value=self.most_occuring_label(y))
        left_side = X[:, best_feature] <= best_threshold
        right_side = X[:, best_feature] > best_threshold
        left_node = self.build_tree(X[left_side], y[left_side])
        right_node = self.build_tree(X[right_side], y[right_side])
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def predict_one_row(self, node, row):
        if Treenode.value is not None:
            return Treenode.value
        if row[node.feature] <= Treenode.threshold:
            return self.predict_one_row(node.left, row)
        return self.predict_one_row(node.right, row)

    def predict(self, X): 
        y_pred = [self.predict_one_row(self.root, row) for row in X]
        return np.array(y_pred)
    
    def cross_validation_accuracy(tree, X, y, k=10):
        kfolds = KFold(k=10, random_state=None, shuffle=True)
        accuracies = []
        for train, test in kf.split(X):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            tree.root = tree.build_tree(X_train, y_train)
            y_pred = tree.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        return np.mean(accuracies), np.std(accuracies)

#Running the code
n_min_values = [5, 10, 15, 20]
for n_min in n_min_values:
    tree = DecisionTree(min_samples_split=n_min)
    mean_acc, std_acc = cross_validation_accuracy(tree, X_iris, y_iris, k=10)
    print(f"Iris - n_min: {n_min}, Accuracy: {mean_acc:.3f}, Std: {std_acc:.3f}")