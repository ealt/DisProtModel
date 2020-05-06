import numpy as np
import unittest
from models import ModalValueClassifier
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


if __name__ == '__main__':
    unittest.main()
