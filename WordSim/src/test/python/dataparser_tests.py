import unittest

from dataparser import Parser

class DataParserTests(unittest.TestCase):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.parser = Parser("../resources/train.wtag", "../resources/test_sample.wtag", 6, False)
        
    def test_num_of_words(self):
        parser = Parser('../resources/SimLex-999.csv', '../resources/wordsim353.csv')
#         self.assertEqual(len(parser.wordsim_db), 351)
#         self.assertEqual(len(parser.simlex_db), 999)
        self.assertEqual(parser.wordsim_db[frozenset({'tiger', 'cat'})].score, 7.35)
    

if __name__ == "__main__":
    unittest.main()