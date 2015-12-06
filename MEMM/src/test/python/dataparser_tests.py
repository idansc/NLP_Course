import unittest
import constants

from dataparser import Parser

class DataParserTests(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = Parser("../resources/train.wtag", "../resources/test_sample.wtag", 6, False)
        
    def test_num_of_words(self):
        self.assertEqual(self.parser.get_num_words(), 121815)
        
        parser = Parser("../resources/train_sample.wtag", "../resources/test_sample.wtag", 6, False)
        self.assertEqual(parser.get_num_words(), 49)
    
    def test_get_all_tags(self):
        self.assertEqual(self.parser.get_all_tags(), constants.TAGS)
    
if __name__ == "__main__":
    unittest.main()