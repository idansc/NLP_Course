import unittest
import utils

from dataparser import Parser

class DataParserTests(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = Parser("../resources/train.wtag")
        
    def test_num_of_words(self):
        self.assertEqual(self.parser.get_num_words(), 121815)
        
        parser = Parser("../resources/sample.wtag")
        self.assertEqual(parser.get_num_words(), 49)
    
    def test_get_all_tags(self):
        self.assertEqual(self.parser.get_all_tags(), utils.TAGS)
    
if __name__ == "__main__":
    unittest.main()