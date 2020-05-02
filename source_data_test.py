import numpy as np
import unittest
from unittest.mock import MagicMock
from source_data import SourceData


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


if __name__ == '__main__':
    unittest.main()