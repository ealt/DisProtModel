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
        if isinstance(Y, np.ndarray):
            self._mode = stats.mode(np.concatenate(Y))[0][0]
        else:
            unigram_counts = Counter()
            for y in utils.flatten(Y):
                unigram_counts[y] += 1
            for val in ignore:
                del unigram_counts[val]
            self._mode = max(unigram_counts.items(), key=itemgetter(1))[0]
        return self

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
        unigram_counts = Counter()
        pair_counts = defaultdict(Counter)
        for x, y in zip(utils.flatten(X), utils.flatten(Y)):
            if y not in self._ignore:
                unigram_counts[y] += 1
                pair_counts[x][y] += 1
        self._mode = max(unigram_counts.items(), key=itemgetter(1))[0]
        self._most_likely_y = {x: y_counts.most_common(1)[0][0]
                               for x, y_counts in pair_counts.items()}
        return self
    
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
