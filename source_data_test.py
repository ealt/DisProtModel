import unittest
import source_data


class SourceDataTest(unittest.TestCase):
    
    def test_get_ids(self):
        ids = source_data.get_ids()
        self.assertTrue(type(ids), list)
        self.assertGreater(len(ids), 0)
        self.assertTrue(all([type(id) == unicode for id in ids]))


if __name__ == '__main__':
    unittest.main()