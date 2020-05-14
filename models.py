from collections import defaultdict, Counter
import numpy as np
from operator import itemgetter
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import utils

class ModalValueClassifier(BaseEstimator,ClassifierMixin):
    def fit(self, X, Y=None, ignore=[], **kwargs):
        if Y is None:
            Y = X
        self._ignore = set(ignore)
        self._mode = self._get_mode(Y)
        return self
    
    def _get_mode(self, Y):
        try:
            return stats.mode([y for y in np.concatenate(Y)
                            if y not in self._ignore], nan_policy='omit')[0][0]
        except:
            return stats.mode([y for y in utils.flatten(Y)
                            if y not in self._ignore], nan_policy='omit')[0][0]

    def predict(self, X):
        if not hasattr(self, '_mode'):
            raise RuntimeError("Model must be fit before calling predict")
        if isinstance(X, np.ndarray):
            return np.full_like(X, self._mode, dtype=type(self._mode))
        else:
            return self._get_predictions(X)
    
    def _get_predictions(self, X):
        if hasattr(X, '__iter__') and type(X) != str:
            if isinstance(X, np.ndarray):
                return np.full_like(X, self._mode, dtype=type(self._mode))
            else:
                return [self._get_predictions(x) for x in X]
        else:
            return self._mode

    def score(self, X, Y=None):
        if not hasattr(self, '_mode'):
            raise RuntimeError("Model must be fit before calling score")
        if Y is None:
            Y = X
        correct = 0
        total = 0
        for y in utils.flatten(Y):
            if not y in self._ignore:
                total += 1
                if y == self._mode:
                    correct += 1
        assert(total > 0)
        return correct / total


class NaiveClassifier(BaseEstimator,ClassifierMixin):
    def fit(self, X, Y, ignore=[], **kwargs):
        self._ignore = set(ignore)
        self._mode = self._get_mode(Y)
        self._most_likely_y = self._get_most_likely_y(X, Y)
        return self
    
    def _get_mode(self, Y):
        try:
            return stats.mode([y for y in np.concatenate(Y)
                            if y not in self._ignore], nan_policy='omit')[0][0]
        except:
            return stats.mode([y for y in utils.flatten(Y)
                            if y not in self._ignore], nan_policy='omit')[0][0]

    def _get_most_likely_y(self, X, Y):
        try:
            pairs, pair_counts = np.unique([[x, y]
                        for x, y in zip(np.concatenate(X), np.concatenate(Y))
                        if y not in self._ignore], axis=0, return_counts=True)
        except:
            pairs, pair_counts = np.unique([[x, y]
                        for x, y in zip(utils.flatten(X), utils.flatten(Y))
                        if y not in self._ignore], axis=0, return_counts=True)
        count_dict = defaultdict(Counter)
        for pair, pair_count in zip(pairs, pair_counts):
            count_dict[pair[0]][pair[1]] = pair_count
        return {x: y_counts.most_common(1)[0][0]
                for x, y_counts in count_dict.items()}
    
    def predict(self, X):
        if not (hasattr(self, '_mode') and hasattr(self, '_most_likely_y')):
            raise RuntimeError("Model must be fit before calling predict")
        return np.array([self._most_likely_y.get(x, self._mode)
                         for x in utils.flatten(X)])

    def score(self, X, Y):
        if not (hasattr(self, '_mode') and hasattr(self, '_most_likely_y')):
            raise RuntimeError("Model must be fit before calling score")
        correct = 0
        total = 0
        for x, y in zip(utils.flatten(X), utils.flatten(Y)):
            if y not in self._ignore:
                total += 1
                if self._most_likely_y.get(x, self._mode) == y:
                    correct += 1
        assert(total > 0)
        return correct / total
