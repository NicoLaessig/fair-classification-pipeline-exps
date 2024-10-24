import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from itertools import combinations
from scipy.special import gammaln
from .functions import discretization

def _roc_auc_score(y, p):
    return 1 - roc_auc_score(y, p)

def brier_score(y, p):
    return mean_squared_error(y, p)


def cross_entropy_loss(y, p):
    eps = 1e-6
    p = np.clip(p, a_min=eps, a_max=1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def poisson_loss(y, p):
    return -np.mean(y * np.log(p) - p - gammaln(y + 1))


def misclassification_rate(y_true, y_pred):
    return 1. - accuracy_score(
        y_true.argmax(1) if y_true.ndim == 2 else y_true,
        y_pred.argmax(1) if y_pred.ndim == 2 else (y_pred > 0.5) * 1.)


class Evaluator:
    def __init__(self, family='bernoulli'):
        self.family = family
        if family == 'bernoulli':
            self.metric_names = [
                'auroc',
                'misclassification_rate',
                'negative_log_likelihood',
                'brier_score'
            ]
            self.metric_functions = [
                _roc_auc_score,
                misclassification_rate,
                cross_entropy_loss,
                brier_score
            ]

        elif family == 'normal':
            self.metric_names = [
                'mean_squared_error',
                'mean_absolute_error',
                'negative_log_likelihood']
            self.metric_functions = [
                mean_squared_error,
                mean_absolute_error,
                mean_squared_error
            ]

        elif family == 'poisson':
            self.metric_names = [
                'mean_squared_error',
                'mean_absolute_error',
                'negative_log_likelihood']

            self.metric_functions = [
                mean_squared_error,
                mean_absolute_error,
                poisson_loss
            ]

        elif family == 'multinomial':
            self.metric_names = [
                'brier_score',
                'misclassification_error',
                'negative_log_likelihood'
            ]

            self.metric_functions = [
                mean_squared_error,
                misclassification_rate,
                cross_entropy_loss
            ]

        # self.disparity_names = ['ΔDP', 'ΔEO'] + [f'cΔ{name}' for name in self.metric_names] + [f'Δ{name}' for name in self.metric_names]
        self.disparity_names = [f'Δ{name}' for name in self.metric_names] + ['ΔEO']

    def evaluate(self, y, p, A):
        res = {}

        for name, metric_fn in zip(self.metric_names, self.metric_functions):
            res[name] = metric_fn(y, p)
            for a in set(A):
                res[f'group{str(int(a) + 1)}-{name}'] = metric_fn(y[A == a], p[A == a])

            res[f'Δ{name}'] = self.disparity(y, p, A, metric_fn)

        res['ΔEO'] = self.compute_deo(y, p, A)
        return res

    def compute_deo(self, y, p, A):
        yd = discretization(y, A, family=self.family, max_segments=100, discretization='equal_length')

        stack = []
        for yy in set(yd):
            yy_stack = []
            for a, b in combinations(set(A), 2):
                Iay = np.logical_and(A == a, yd == yy)
                Iby = np.logical_and(A == b, yd == yy)
                if sum(Iay) > 0 and sum(Iby) > 0:
                    if self.family != 'multinomial':
                        yy_stack.append(
                            np.square(np.subtract(p[Iay].mean(), p[Iby].mean())))
                    else:
                        yy_stack.append(
                            np.square(p[Iay].mean(0) - p[Iby].mean(0)).sum()
                        )
            if len(yy_stack) > 0:
                stack.append(np.mean(yy_stack))

        return np.sqrt(np.mean(stack))

    def disparity(self, y, p, A, metric_fn):
        stack = []
        if metric_fn == _roc_auc_score:
            for a, b in combinations(set(A), 2):
                stack.append(
                    np.abs(
                        np.subtract(
                            metric_fn(y[A == a], p[A == a]),
                            metric_fn(y[A == b], p[A == b])
                        )))

        else:
            yd = discretization(y, A, family=self.family, max_segments=100, discretization='equal_length')
            for yy in set(yd):
                yy_stack = []
                for a, b in combinations(set(A), 2):
                    Iay = np.logical_and(A == a, yd == yy)
                    Iby = np.logical_and(A == b, yd == yy)
                    yy_stack.append(
                        np.square(
                            np.subtract(
                                metric_fn(y[Iay], p[Iay]),
                                metric_fn(y[Iby], p[Iby])
                            )))
                stack.append(np.mean(yy_stack))
        return np.sqrt(np.mean(stack))
