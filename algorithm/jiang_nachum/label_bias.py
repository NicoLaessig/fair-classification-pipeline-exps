import numpy as np

class LabelBiasDP_helper():
    def __init__(self):
        pass

    def debias_weights(self, original_labels, protected_attributes, multipliers):
        exponents = np.zeros(len(original_labels))
        for i, m in enumerate(multipliers):
            exponents -= m * protected_attributes[i]
        weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
        weights = np.where(original_labels > 0, 1 - weights, weights)
        return weights

    def get_error_and_violations(self, y_pred, y, protected_attributes):
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(p > 0)
            violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        pairwise_violations = []
        for i in range(len(protected_attributes)):
            for j in range(i+1, len(protected_attributes)):
                protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
                if len(protected_idxs[0]) == 0:
                    continue
                pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        return acc, violations, pairwise_violations



class LabelBiasEOD_helper():
    def __init__(self):
        pass

    def debias_weights(self, original_labels, predicted, protected_attributes, multipliers):
        exponents_pos = np.zeros(len(original_labels))
        exponents_neg = np.zeros(len(original_labels))

        for i, protected in enumerate(protected_attributes):
            exponents_pos -= multipliers[2 * i] * protected
            exponents_neg -= multipliers[2 * i + 1] * protected
        weights_pos = np.exp(exponents_pos)/ (np.exp(exponents_pos) + np.exp(-exponents_pos))
        weights_neg = np.exp(exponents_neg)/ (np.exp(exponents_neg) + np.exp(-exponents_neg))

        #weights = np.where(predicted > 0, weights, 1 - weights)
        weights = np.where(original_labels > 0, 1 - weights_pos, weights_neg)
        return weights


    def get_error_and_violations(self, y_pred, y, protected_attributes):
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0))
            positive_idxs = np.where(y > 0)
            violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
            protected_idxs = np.where(np.logical_and(p > 0, y < 1))
            negative_idxs = np.where(y < 1)
            violations.append(np.mean(y_pred[negative_idxs]) - np.mean(y_pred[protected_idxs]))
        pairwise_violations = []
        for i in range(len(protected_attributes)):
            for j in range(i+1, len(protected_attributes)):
                protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
                if len(protected_idxs[0]) == 0:
                    continue
                pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        return acc, violations, pairwise_violations



class LabelBiasEOP_helper():
    def __init__(self):
        pass

    def debias_weights(self, original_labels, predicted, protected_attributes, multipliers):
        exponents = np.zeros(len(original_labels))
        for i, m in enumerate(multipliers):
            exponents -= m * protected_attributes[i]
        weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
        #weights = np.where(predicted > 0, weights, 1 - weights)
        weights = np.where(original_labels > 0, 1 - weights, weights)
        return weights

    def get_error_and_violations(self, y_pred, y, protected_attributes):
        acc = np.mean(y_pred != y)
        violations = []
        for p in protected_attributes:
            protected_idxs = np.where(np.logical_and(p > 0, y > 0))
            positive_idxs = np.where(y > 0)
            violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
        pairwise_violations = []
        for i in range(len(protected_attributes)):
            for j in range(i+1, len(protected_attributes)):
                protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
                if len(protected_idxs[0]) == 0:
                    continue
                pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
        return acc, violations, pairwise_violations
