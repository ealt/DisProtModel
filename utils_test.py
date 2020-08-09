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
        expected_total = {0.: 8, 1.: 3}

        # k = 1
        expected_head = {0.: 7, 1.: 2}
        expected_tail = {0.: 6, 1.: 3}
        total, head, tail = utils.get_value_counts(X)
        self.assertDictEqual(total, expected_total)
        self.assertDictEqual(head, expected_head)
        self.assertDictEqual(tail, expected_tail)

        k = 4
        expected_head = {0.: 3, 1.: 1}
        expected_tail = {0.: 2, 1.: 2}
        total, head, tail = utils.get_value_counts(X, k)
        self.assertDictEqual(total, expected_total)
        self.assertDictEqual(head, expected_head)
        self.assertDictEqual(tail, expected_tail)

        k = 7
        expected_total = {0.: 3, 1.: 3}
        expected_head = {0.: 1}
        expected_tail = {1.: 1}
        total, head, tail = utils.get_value_counts(X, k)
        self.assertDictEqual(total, expected_total)
        self.assertDictEqual(head, expected_head)
        self.assertDictEqual(tail, expected_tail)

    def test_get_frequency_inner_product(self):
        counts_1 = {0.: 1, 1.: 3}  # freq = [1/4, 3/4, 0]

        counts_2 = {0.: 2, 1.: 4, 2.: 2}  # freq = [1/4, 1/2, 1/4]        
        expected_frequency_inner_product = 0.4375  # 7/16
        self.assertEqual(utils.get_frequency_inner_product(counts_1, counts_2),
                         expected_frequency_inner_product)

        counts_3 = {2.: 3}  # freq = [0, 0, 1]
        expected_frequency_inner_product = 0.
        self.assertEqual(utils.get_frequency_inner_product(counts_1, counts_3),
                         expected_frequency_inner_product)


if __name__ == '__main__':
    unittest.main()
