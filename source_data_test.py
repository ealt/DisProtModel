from tempfile import TemporaryFile
from mock import patch
import numpy as np
from sklearn.model_selection import train_test_split
import unittest
from unittest.mock import MagicMock
from source_data import SourceData


MOCK_IDS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
MOCK_TRAIN_IDS, MOCK_TEST_IDS = train_test_split(MOCK_IDS,
                                                 test_size=0.2, random_state=42)
MOCK_DATA = {
    '0': {
        'X': np.array(['A', 'B', 'C', 'D', 'E'], dtype=np.unicode_),
        'Y': np.array(['N', 'O', 'P', 'Q', 'R'], dtype=np.unicode_)
    },
    '1': {
        'X': np.array(['B', 'C', 'D', 'E', 'F'], dtype=np.unicode_),
        'Y': np.array(['O', 'P', 'Q', 'R', 'S'], dtype=np.unicode_)
    },
    '2': {
        'X': np.array(['C', 'D', 'E', 'F', 'G'], dtype=np.unicode_),
        'Y': np.array(['P', 'Q', 'R', 'S', 'T'], dtype=np.unicode_)
    },
    '3': {
        'X': np.array(['D', 'E', 'F', 'G', 'H'], dtype=np.unicode_),
        'Y': np.array(['Q', 'R', 'S', 'T', 'U'], dtype=np.unicode_)
    },
    '4': {
        'X': np.array(['E', 'F', 'G', 'H', 'I'], dtype=np.unicode_),
        'Y': np.array(['R', 'S', 'T', 'U', 'V'], dtype=np.unicode_)
    },
    '5': {
        'X': np.array(['F', 'G', 'H', 'I', 'J'], dtype=np.unicode_),
        'Y': np.array(['S', 'T', 'U', 'V', 'W'], dtype=np.unicode_)
    },
    '6': {
        'X': np.array(['G', 'H', 'I', 'J', 'K'], dtype=np.unicode_),
        'Y': np.array(['T', 'U', 'V', 'W', 'X'], dtype=np.unicode_)
    },
    '7': {
        'X': np.array(['H', 'I', 'J', 'K', 'L'], dtype=np.unicode_),
        'Y': np.array(['U', 'V', 'W', 'X', 'Y'], dtype=np.unicode_)
    },
    '8': {
        'X': np.array(['I', 'J', 'K', 'L', 'M'], dtype=np.unicode_),
        'Y': np.array(['V', 'W', 'X', 'Y', 'Z'], dtype=np.unicode_)
    },
    '9': {
        'X': np.array(['J', 'K', 'L', 'M', 'N'], dtype=np.unicode_),
        'Y': np.array(['W', 'X', 'Y', 'Z', '['], dtype=np.unicode_)
    }
}
def get_data_mock(id):
    return MOCK_DATA[id]['X'], MOCK_DATA[id]['Y']


class SourceDataTest(unittest.TestCase):
    
    def test_get_ids(self):
        ids = SourceData.get_ids()
        self.assertTrue(type(ids), list)
        self.assertGreater(len(ids), 0)
        self.assertTrue(all([type(id) == str for id in ids]))

    def test_get_data(self):
        id = 'DP00086'
        expected_X = np.array([
            77, 69, 69, 80, 81, 83, 68, 80, 83, 86, 69, 80, 80, 76, 83, 81, 69,
            84, 70, 83, 68, 76, 87, 75, 76, 76, 80, 69, 78, 78, 86, 76, 83, 80,
            76, 80, 83, 81, 65, 77, 68, 68, 76, 77, 76, 83, 80, 68, 68, 73, 69,
            81, 87, 70, 84, 69, 68, 80, 71, 80, 68, 69, 65, 80, 82, 77, 80, 69,
            65, 65, 80, 80, 86, 65, 80, 65, 80, 65, 65, 80, 84, 80, 65, 65, 80,
            65, 80, 65, 80, 83, 87, 80, 76, 83, 83, 83, 86, 80, 83, 81, 75, 84,
            89, 81, 71, 83, 89, 71, 70, 82, 76, 71, 70, 76, 72, 83, 71, 84, 65,
            75, 83, 86, 84, 67, 84, 89, 83, 80, 65, 76, 78, 75, 77, 70, 67, 81,
            76, 65, 75, 84, 67, 80, 86, 81, 76, 87, 86, 68, 83, 84, 80, 80, 80,
            71, 84, 82, 86, 82, 65, 77, 65, 73, 89, 75, 81, 83, 81, 72, 77, 84,
            69, 86, 86, 82, 82, 67, 80, 72, 72, 69, 82, 67, 83, 68, 83, 68, 71,
            76, 65, 80, 80, 81, 72, 76, 73, 82, 86, 69, 71, 78, 76, 82, 86, 69,
            89, 76, 68, 68, 82, 78, 84, 70, 82, 72, 83, 86, 86, 86, 80, 89, 69,
            80, 80, 69, 86, 71, 83, 68, 67, 84, 84, 73, 72, 89, 78, 89, 77, 67,
            78, 83, 83, 67, 77, 71, 71, 77, 78, 82, 82, 80, 73, 76, 84, 73, 73,
            84, 76, 69, 68, 83, 83, 71, 78, 76, 76, 71, 82, 78, 83, 70, 69, 86,
            82, 86, 67, 65, 67, 80, 71, 82, 68, 82, 82, 84, 69, 69, 69, 78, 76,
            82, 75, 75, 71, 69, 80, 72, 72, 69, 76, 80, 80, 71, 83, 84, 75, 82,
            65, 76, 80, 78, 78, 84, 83, 83, 83, 80, 81, 80, 75, 75, 75, 80, 76,
            68, 71, 69, 89, 70, 84, 76, 81, 73, 82, 71, 82, 69, 82, 70, 69, 77,
            70, 82, 69, 76, 78, 69, 65, 76, 69, 76, 75, 68, 65, 81, 65, 71, 75,
            69, 80, 71, 71, 83, 82, 65, 72, 83, 83, 72, 76, 75, 83, 75, 75, 71,
            81, 83, 84, 83, 82, 72, 75, 75, 76, 77, 70, 75, 84, 69, 71, 80, 68,
            83, 68])
        expected_Y = np.array([
            84., 84., 84., 84., 84., 84., 84., 84., 84., 84., 84., 84., 84.,
            84., 84., 84., 84., 84., 84., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 84., 84., 84., 84., 84.,
            84., 84., 84., 84., 84., 84., 84., 84., 84., 84., 84., 84., 84.,
            84., 84., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73.,
            73., 73., 73., 73., 73., 73., 68., 68., 68., 68., 68., 68., 68.,
            68., 68., 68., 68., 68., 68., 68., 68., 68., 68., 68., 68., 68.,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 73., 73., 
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 
            73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 73., 
            73., 73., 73., 73., 73., 73., 73.])
        X, Y = SourceData.get_data(id)
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(Y, expected_Y)

    def test_load_data_sets(self):
        expected_data = [
            np.array([MOCK_DATA[id]['X'] for id in MOCK_TRAIN_IDS]),
            np.array([MOCK_DATA[id]['X'] for id in MOCK_TEST_IDS]),
            np.array([MOCK_DATA[id]['Y'] for id in MOCK_TRAIN_IDS]),
            np.array([MOCK_DATA[id]['Y'] for id in MOCK_TEST_IDS]),
        ]
        data = SourceData.load_data_sets('test_data.npz')
        for data_set, expected_data_set in zip(data, expected_data):
            self.assertTrue(np.array_equal(data_set, expected_data_set))


    @patch.object(SourceData, 'get_ids', return_value=MOCK_IDS)
    @patch.object(SourceData, 'get_data', side_effect=get_data_mock)
    def test_get_data_sets(self, mock_get_ids, mock_get_data):
        expected_data = [
            [MOCK_DATA[id]['X'] for id in MOCK_TRAIN_IDS],
            [MOCK_DATA[id]['X'] for id in MOCK_TEST_IDS],
            [MOCK_DATA[id]['Y'] for id in MOCK_TRAIN_IDS],
            [MOCK_DATA[id]['Y'] for id in MOCK_TEST_IDS],
        ]
        data = SourceData.load_data_sets('nonexistant_file.npz')
        self.assertListEqual(data, expected_data)
    
    def test_save_data_sets(self):
        X_train = [MOCK_DATA[id]['X'] for id in MOCK_TRAIN_IDS]
        X_test  = [MOCK_DATA[id]['X'] for id in MOCK_TEST_IDS]
        Y_train = [MOCK_DATA[id]['Y'] for id in MOCK_TRAIN_IDS]
        Y_test  = [MOCK_DATA[id]['Y'] for id in MOCK_TEST_IDS]
        with TemporaryFile() as outfile:
            SourceData.save_data_sets(X_train=X_train, X_test=X_test,
                                    Y_train=Y_train, Y_test=Y_test,
                                    file_args=outfile)
            _ = outfile.seek(0)  # Needed to simulate closing & reopening file
            with np.load(outfile) as infile:
                self.assertTrue(np.array_equal(infile['X_train'], X_train))
                self.assertTrue(np.array_equal(infile['X_test'],  X_test))
                self.assertTrue(np.array_equal(infile['Y_train'], Y_train))
                self.assertTrue(np.array_equal(infile['Y_test'],  Y_test))


if __name__ == '__main__':
    unittest.main()