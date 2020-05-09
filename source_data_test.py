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
        expected_X = np.array(list(
            'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRM'
            'PEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNK'
            'MFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVE'
            'GNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNL'
            'LGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTL'
            'QIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'),
            dtype=np.unicode_)
        expected_Y = np.array(list(
            'TTTTTTTTTTTTTTTTTTTIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            'IIIIIIITTTTTTTTTTTTTTTTTTTTIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            'IIIIIIIIIIIIIIIIIIIIIIIIIIIIDDDDDDDDDDDDDDDDDDDD???????IIIIIIIIIII'
            'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'),
            dtype=np.unicode_)
        X, Y = SourceData.get_data(id)
        self.assertTrue(np.array_equal(X, expected_X))
        self.assertTrue(np.array_equal(Y, expected_Y))

    @patch.object(SourceData, 'get_ids', return_value=MOCK_IDS)
    @patch.object(SourceData, 'get_data', side_effect=get_data_mock)
    def test_get_data_sets(self, mock_get_ids, mock_get_data):
        expected_data = (
            [MOCK_DATA[id]['X'] for id in MOCK_TRAIN_IDS],
            [MOCK_DATA[id]['X'] for id in MOCK_TEST_IDS],
            [MOCK_DATA[id]['Y'] for id in MOCK_TRAIN_IDS],
            [MOCK_DATA[id]['Y'] for id in MOCK_TEST_IDS],
        )
        data = SourceData.load_data_sets('nonexistant_file.npz')
        self.assertListEqual(data, expected_data)


if __name__ == '__main__':
    unittest.main()