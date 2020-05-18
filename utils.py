from collections import Counter
import numpy as np
import requests

def try_get_request(kwargs):
    try:
        response = requests.get(**kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(e)

def flatten(obj):
    if type(obj) == str or not hasattr(obj, '__iter__'):
        yield obj
    else:
        for element in obj:
            yield from flatten(element)

def get_lag_match_frequency(X, k=1):
    num_matches = 0
    num_total = 0
    for x in X:
        dx = x[k:] - x[:-k]
        n = dx.size
        num_matches += n - np.count_nonzero(dx)
        num_total += n - np.count_nonzero(np.isnan(dx))
    return num_matches / num_total

def get_value_counts(X, k=1):
    head = Counter()
    mid = Counter()
    tail = Counter()
    for x in X:
        if len(x) >= 2 * k:
            update_counts(head, x[:k])
            update_counts(mid, x[k:-k])
            update_counts(tail, x[-k:])
        else:
            update_counts(head, x[:-k])
            update_counts(tail, x[k:])
    head += mid
    tail += mid
    return head, tail

def update_counts(counter, x):
    values, value_counts = np.unique(x, return_counts=True)
    for value, value_count in zip(values, value_counts):
        if ~np.isnan(value):
            counter[value] += value_count

def eq(a, b):
    if not (hasattr(a, '__iter__') or type(a) == str):
        return a == b
    try:
        if not len(a) == len(b):
            return False
        elif isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        elif isinstance(a, dict):
            return all(eq(v, b[k]) for k, v in a.items())
        elif isinstance(a, set):
            return a.symmetric_difference(b) == set()
        else:
            return all(eq(a_i, b_i) for a_i, b_i in zip(a, b))
    except (TypeError, KeyError):
        return False
