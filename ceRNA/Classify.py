import pdb

import numpy as np


def classify_nodes_as_in_network(x, y, mRNA, miRNA, tests, wild_type_test, changes):
    n = len(tests)
    fake_x = x + mRNA

    node_class = []
    classification_accuracy = {"True Positive": 0, "False Negative": 0, "True Negative": 0, "False Positive": 0}
    count_correct = np.zeros(n)
    count_incorrect = np.zeros(n)
    for i in range(n):
        for j in range(0, n):
            changed = changes[j % len(changes)]
            if i < fake_x:
                if changed < fake_x:
                    c = 1
                else:
                    c = -1
            else:
                if changed < fake_x:
                    c = -1
                else:
                    c = 1
            if c * (tests[j][i] - wild_type_test[i]) > 0:
                count_correct[i] += 1
            else:
                count_incorrect[i] += 1

    for i in range(n):
        if count_correct[i] / float(n) >= .70:
            node_class.append(True)
            if i in range(x) or i in range(fake_x, fake_x + y):
                classification_accuracy["True Positive"] += 1
            else:
                classification_accuracy["False Positive"] += 1
        else:
            node_class.append(False)
            if i in range(x) or i in range(fake_x, fake_x + y):
                classification_accuracy["False Negative"] += 1
            else:
                classification_accuracy["True Negative"] += 1
    return classification_accuracy, node_class


def classify_model_as_incomplete(singular_values):
    s = len(singular_values)
    for i in range(s):
        if singular_values[i] <= 1e-8:
            return True
    return False


def classify_full_nullspace_as_incorrect(x, y, prediction):
    n = x + y
    for i in range(x):
        pdb.set_trace()
        if not (prediction[i] > 0):
            return False
        if not (prediction[n + i] < 0):
            return False
    for i in range(y):
        if not (prediction[x + i] < 0):
            return False
        if not (prediction[n + x + i] > 0):
            return False
    return True


def classify_half_nullspace_as_incorrect(x, y, prediction):
    for i in range(x):
        if prediction[i] <= 0:
            return False
    for i in range(y):
        if prediction[x + i] >= 0:
            return False
    return True
