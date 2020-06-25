# Utility function for tree based models.
import numpy as np


def entropy(y):
    total = y.shape[0]
    _, count = np.unique(y, return_counts=True)

    p = count / total
    ent = np.sum(-p*np.log(p))
    return ent


def gini_impurity(y):
    total = y.shape[0]
    _, count = np.unique(y, return_counts=True)

    p = count / total
    gini = np.sum(p * (1 - p))
    return gini


def split_on_criteria(X, feature, criteria):
    """
    Split the original dataset into two subsets based on condition.
    Args:
        dataset: Original dataset.
        feature: Feature to split on.
        criteria: Condition of the feature to split on.

    Returns: Two separate datasets based on condition.

    """
    if isinstance(criteria, (int, float)):
        true_indexes = X[X[feature] >= criteria].index
        false_indexes = X[X[feature] < criteria].index
    else:
        true_indexes = X[X[feature] == criteria].index
        false_indexes = X[X[feature] != criteria].index
    return true_indexes, false_indexes


def print_tree(tree, indent='', level=1, t_or_f=''):
    if not tree:
        return
    if tree.result is not None:
        print(indent, '--', "Leaf:", t_or_f, tree.result[0][0], 'Score:', tree.score)
    else:
        print(indent, '--', level, t_or_f, tree, 'Score:', tree.score)

    print_tree(tree.true_node, indent=indent + '    ', level=level + 1, t_or_f='T')
    print_tree(tree.false_node, indent=indent + '    ', level=level + 1, t_or_f='F')

