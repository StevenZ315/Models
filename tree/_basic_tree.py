import numpy as np
import pandas as pd
from utils import entropy, gini_impurity, split_on_criteria, print_tree


class TreeNode:
    def __init__(self, feature=None, criteria=None, result=None, true_node=None, false_node=None, score=None):
        self.feature = feature   # Feature to split on.
        self.criteria = criteria    # Split criteria. (>=criteria for int&float. ==criteria for else)
        self.result = result    # Split result in a dict structure.
        self.true_node = true_node  # Child node where split condition is met.
        self.false_node = false_node    # Child node where split condition is not met.
        self.score = score

    def __str__(self):
        if isinstance(self.criteria, (int, float)):
            info = str(self.feature) + ' >= ' + str(self.criteria) + '?'
        else:
            info = str(self.feature) + ' is ' + str(self.criteria) + '?'
        return info


class DecisionTreeClassifier:
    def __init__(self, criterion='gini',
                 min_impurity_decrease=0.0,
                 pruning_alpha=0.0,
                 min_samples_split=2):
        if criterion == 'gini':     # Set scoring function.
            self.score_func = gini_impurity
        elif criterion == 'entropy':
            self.score_func = entropy
        else:
            raise AttributeError('Invalid criterion input: ', criterion)

        self.min_impurity_decrease = min_impurity_decrease  # Threshold for early stopping in tree growth.
        self.pruning_alpha = pruning_alpha     # Minimum cost-complexity reduction factor.
        self.min_samples_split = min_samples_split

        self.tree_structure = TreeNode()

    def fit(self, X, y):
        df_X = pd.DataFrame.from_records(X)
        self.tree_structure = self.build_tree(df_X, y)
        if self.pruning_alpha > 0:
            self.prune(self.tree_structure)

    def predict(self, X):
        prediction = [self._predict(x, self.tree_structure) for x in X]
        return prediction

    def _predict(self, x, node):
        if node.result is not None:
            return node.result[0][0]
        else:
            value = x[node.feature]
            criteria = node.criteria
            if isinstance(criteria, (int, float)):
                if value >= criteria:
                    child = node.true_node
                else:
                    child = node.false_node
            else:
                if value == criteria:
                    child = node.true_node
                else:
                    child = node.false_node
            return self._predict(x, child)

    def build_tree(self, X, y):
        """
        Build the decision tree.
        Args:
            X:
            y:

        Returns:

        """
        if len(X) == 0:
            return None
        curr_score = self.score_func(y)
        if len(X) < self.min_samples_split:     # Return a leaf node contains only results.
            unique_count = np.unique(y, return_counts=True)
            result = [(unique_count[0][idx], unique_count[1][idx]) for idx in range(len(unique_count[0]))]
            result.sort(key=lambda x: x[1], reverse=True)
            return TreeNode(result=result, score=curr_score)

        best_set_indexes, best_criteria, best_gain = self.best_split(X, y)
        if best_gain <= 0:
            return TreeNode(result=[(y[0], len(y))], score=curr_score)
        else:
            true_node = self.build_tree(X.loc[best_set_indexes[0]].reset_index(drop=True), y[best_set_indexes[0]])
            false_node = self.build_tree(X.loc[best_set_indexes[1]].reset_index(drop=True), y[best_set_indexes[1]])
            return TreeNode(feature=best_criteria[0], criteria=best_criteria[1],
                            true_node=true_node, false_node=false_node, score=curr_score)

    def best_split(self, X, y):
        """
        Find the best feature & criteria to split.
        Args:
            X:
            y:

        Returns:

        """
        best_set_indexes, best_criteria, best_gain = None, None, 0.0
        curr_score = self.score_func(y)

        for column in X.columns:     # Iterate over all features
            criteria = X[column].unique()
            for value in criteria:      # Iterate over all criteria
                true_indexes, false_indexes = split_on_criteria(X, column, value)

                # Calculate information gain.
                p = len(true_indexes) / len(y)
                score_new = p * self.score_func(y[true_indexes]) + (1 - p) * self.score_func(y[false_indexes])
                gain = curr_score - score_new

                if gain > best_gain:
                    best_set_indexes, best_criteria, best_gain = (true_indexes, false_indexes), (column, value), gain

        return best_set_indexes, best_criteria, best_gain

    def prune(self, tree):
        # Prune the child node if it's not leaf node.
        if tree.true_node.result is None:
            self.prune(tree.true_node)
        if tree.false_node.result is None:
            self.prune(tree.false_node)

        if tree.true_node.result is not None and tree.false_node.result is not None:
            # Compare the
            delta = tree.score - (tree.true_node.score + tree.false_node.score) / 2
            if delta < self.pruning_alpha:
                # Update new result
                y = []
                for value, count in tree.true_node.result:
                    y += [value] * count
                y = np.array(y)
                unique_count = np.unique(y, return_counts=True)
                result = [(unique_count[0][idx], unique_count[1][idx]) for idx in range(len(unique_count[0]))]
                result.sort(key=lambda x: x[1], reverse=True)
                tree.result = result

                # Prune
                tree.true_node = None
                tree.false_node = None

