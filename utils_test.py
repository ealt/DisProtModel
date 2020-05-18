import numpy as np
import unittest
import utils


class UtilsTest(unittest.TestCase):

    def test_flatten(self):
        array = np.array([[[0, 1], [2, 3], [4, 5], [6, 7]],
                          [[0, 0], [2, 3], [4, 5], [6, 7]]], dtype=int)
        array_flattened = [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 2, 3, 4, 5, 6, 7]
        self.assertListEqual(list(utils.flatten(array)), array_flattened)

        string_list = ['zero', 'zero', 'zero', 'one', 'two', 'three']
        self.assertListEqual(list(utils.flatten(string_list)), string_list)

        mixed = [(np.array([0, 1], dtype=int), np.array([2, 3], dtype=int),
                  np.array([4, 5], dtype=int)), [6, 7], (0, 0, 2, 3),
                  np.array([[4, 5], [6, 7]], dtype=int)]
        mixed_flattened = [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 2, 3, 4, 5, 6, 7]
        self.assertListEqual(list(utils.flatten(mixed)), mixed_flattened)

    def test_get_lag_match_frequency(self):
        X = [
            np.array([0, 0, 1, np.nan, np.nan, 0, 1, 1]),
            np.array([0, 0, 0, 0, 0])
        ]
        self.assertEqual(utils.get_lag_match_frequency(X), 0.75)
        self.assertEqual(utils.get_lag_match_frequency(X, 4), 1.0)

    def test_get_value_counts(self):
        X = [
            np.array([0, 0, 1, np.nan, np.nan, 0, 1, 1]),
            np.array([0, 0, 0, 0, 0])
        ]

        expected_head = {0.: 7, 1.: 2}
        expected_tail = {0.: 6, 1.: 3}
        head, tail = utils.get_value_counts(X)
        self.assertDictEqual(head, expected_head)
        self.assertDictEqual(tail, expected_tail)

        expected_head = {0.: 3, 1.: 1}
        expected_tail = {0.: 2, 1.: 2}
        head, tail = utils.get_value_counts(X, 4)
        self.assertDictEqual(head, expected_head)
        self.assertDictEqual(tail, expected_tail)


if __name__ == '__main__':
    unittest.main()
