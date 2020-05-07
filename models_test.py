import numpy as np
import unittest
from models import ModalValueClassifier, NaiveClassifier
import utils


class ModalValueClassifierTest(unittest.TestCase):
    
    def test_fit(self):
        X_array = np.ones((2, 2, 2))
        Y_array = np.array([[[0, 1], [2, 3], [4, 5], [6, 7]],
                            [[0, 0], [2, 3], [4, 5], [6, 7]]], dtype=int)
        model_array = ModalValueClassifier().fit(X_array, Y_array)
        self.assertEqual(model_array._mode, 0)

        Y_string = ['zero', 'zero', 'zero', 'one', 'two', 'three']
        model_string = ModalValueClassifier().fit(Y_string)
        self.assertEqual(model_string._mode, 'zero')

        Y_mixed = [(np.array([0, 1], dtype=int), np.array([2, 3], dtype=int),
                    np.array([4, 5], dtype=int)), [6, 7], (0, 0, 2, 3),
                    np.array([[4, 5], [6, 7]], dtype=int)]
        model_mixed = ModalValueClassifier().fit(Y_mixed)
        self.assertEqual(model_mixed._mode, 0)

    def test_predict(self):
        with self.assertRaises(RuntimeError):
            X = np.ones((2, 2, 2))
            ModalValueClassifier().predict(X)
        
        model_int = ModalValueClassifier()
        model_int._mode = 0

        X_array = np.ones((2, 2, 2))
        expected_prediction_array = np.zeros((2, 2, 2), dtype=int)
        self.assertTrue(np.array_equal(model_int.predict(X_array),
                                       expected_prediction_array))

        X_mixed = [(np.array([0, 1], dtype=int), np.array([2, 3], dtype=int),
                    np.array([4, 5], dtype=int)), [6, 7], (0, 0, 2, 3),
                    np.array([[4, 5], [6, 7]], dtype=int)]
        expected_prediction_mixed = [(np.zeros(2, dtype=int),
                                      np.zeros(2, dtype=int),
                                      np.zeros(2, dtype=int)), [0, 0],
                                     [0, 0, 0, 0], np.zeros((2, 2), dtype=int)]
        self.assertTrue(utils.eq(model_int.predict(X_mixed),
                                 expected_prediction_mixed))

        model_string = ModalValueClassifier()
        model_string._mode = 'zero'
        X_string = [i for i in range(5)]
        expected_prediction_string = ['zero', 'zero', 'zero', 'zero', 'zero']
        self.assertListEqual(model_string.predict(X_string),
                             expected_prediction_string)

    def test_score(self):
        with self.assertRaises(RuntimeError):
            Y = np.ones((2, 2, 2))
            ModalValueClassifier().score(Y)

        model_int = ModalValueClassifier()
        model_int._mode = 0

        X_array = np.ones((2, 2, 2))
        Y_array = np.array([[[0, 1], [2, 3], [4, 5], [6, 7]],
                            [[0, 0], [2, 3], [4, 5], [6, 7]]], dtype=int)
        self.assertEqual(model_int.score(X_array, Y_array), 0.1875)

        Y_mixed = [(np.array([0, 1], dtype=int), np.array([2, 3], dtype=int),
                    np.array([4, 5], dtype=int)), [6, 7], (0, 0, 2, 3),
                    np.array([[4, 5], [6, 7]], dtype=int)]
        self.assertEqual(model_int.score(Y_mixed), 0.1875)

        model_string = ModalValueClassifier()
        model_string._mode = 'zero'
        Y_string = [['zero', 'one',  'two', 'three'],
                    ['zero', 'zero', 'two', 'three']]
        self.assertEqual(model_string.score(Y_string), 0.375)


class NaiveClassifierTest(unittest.TestCase):
    
    def test_fit(self):
        X = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'd'])
        Y = np.array(['x', 'x', 'y', 'z', 'z', 'z', 'y', 'z'])
        expected_most_likely_y = {'a': 'x', 'b': 'z', 'c': 'y', 'd': 'z'}
        model = NaiveClassifier().fit(X, Y)
        self.assertEqual(model._mode, 'z')
        self.assertDictEqual(model._most_likely_y, expected_most_likely_y)

    def test_predict(self):
        with self.assertRaises(RuntimeError):
            X = np.ones((2, 2, 2))
            NaiveClassifier().predict(X)
        
        model = NaiveClassifier()
        model._mode = 'w'
        model._most_likely_y = {
            'a': 'x',
            'b': 'y',
            'c': 'z'
        }

        X     = np.array(['z', 'a', 'b', 'b', 'b', 'c', 'd', 'e'])
        Y_hat = np.array(['w', 'x', 'y', 'y', 'y', 'z', 'w', 'w'])
        self.assertTrue(np.array_equal(model.predict(X), Y_hat))

    def test_score(self):
        with self.assertRaises(RuntimeError):
            X = np.ones((2, 2, 2))
            Y = np.ones((2, 2, 2))
            NaiveClassifier().score(X, Y)
        
        model = NaiveClassifier()
        model._mode = 'w'
        model._most_likely_y = {
            'a': 'x',
            'b': 'y',
            'c': 'z'
        }

        X = np.array(['z', 'a', 'b', 'b', 'b', 'c', 'd', 'e'])
        Y = np.array(['q', 'x', 'y', 'y', 'y', 'z', 'q', 'q'])
        self.assertEqual(model.score(X, Y), 0.625)


if __name__ == '__main__':
    unittest.main()
