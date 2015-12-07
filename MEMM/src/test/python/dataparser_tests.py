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
    
    def test_get_prefixes(self):
        parser = Parser("../resources/dataparser_test/test_get_prefixes.wtag", "../resources/test_sample.wtag", 6, False)
        self.assertEqual(parser.get_prefixes(1, 1), {'H', 'v', 't', 'p', 'a', 'w', 'f', 'i', 'c'})
        self.assertEqual(parser.get_prefixes(1, 2), {'p', 'v', 't'})
        self.assertEqual(parser.get_prefixes(3, 1), {'fas', 'pre', 'con', 'vol', 'ver', 'the', 'wil', 'tra', 'pou'})
    
    def test_get_suffixes(self):
        parser = Parser("../resources/dataparser_test/test_get_prefixes.wtag", "../resources/test_sample.wtag", 6, False)
        self.assertEqual(parser.get_suffixes(1, 1), {'y', 'l', 'd', 'a', 's', 'e', 'o', 'n'})
        self.assertEqual(parser.get_suffixes(1, 2), {'e', 'n'})
        self.assertEqual(parser.get_suffixes(3, 1), {'ery', 'und', 'ile', 'cts', 'ion', 'the', 'ade', 'ill', 'nue'})
        
        
if __name__ == "__main__":
    unittest.main()