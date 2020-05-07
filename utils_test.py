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

if __name__ == '__main__':
    unittest.main()
