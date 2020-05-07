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
