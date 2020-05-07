from collections import Counter
import numpy as np
from operator import itemgetter
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import utils

class ModalValueClassifier(BaseEstimator,ClassifierMixin):
    def fit(self, X, Y=None):
        if Y is None:
            Y = X
        if isinstance(Y, np.ndarray):
            self._mode = stats.mode(Y.flatten())[0][0]
        else:
            unigram_counts = Counter()
            for y in utils.flatten(Y):
                unigram_counts[y] += 1
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
            total += 1
            if y == self._mode:
                correct += 1
        return correct / total
