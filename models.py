from collections import Counter
import numpy as np
from operator import itemgetter
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

class ModalValueClassifier(BaseEstimator,ClassifierMixin):
    def fit(self, X, Y=None):
        if Y is None:
            Y = X
        if isinstance(Y, np.ndarray):
            self._mode = stats.mode(Y.flatten())[0][0]
        else:
            self._unigram_counts = Counter()
            self._update_unigram_counts(Y)
            self._mode = max(self._unigram_counts.items(), key=itemgetter(1))[0]
        return self
    
    def _update_unigram_counts(self, Y):
        if hasattr(Y, '__iter__') and type(Y) != str:
            for y in Y:
                self._update_unigram_counts(y)
        else:
            self._unigram_counts[Y] += 1

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
            raise RuntimeError("Model must be fit before calling predict")
        if Y is None:
            Y = X
        self._correct = 0
        self._total = 0
        self._update_score_counts(Y)
        return self._correct / self._total

    def _update_score_counts(self, Y):
        if hasattr(Y, '__iter__') and type(Y) != str:
            for y in Y:
                self._update_score_counts(y)
        else:
            self._total += 1
            if Y == self._mode:
                self._correct += 1
