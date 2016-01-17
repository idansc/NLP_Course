import unittest

from dataparser import Parser

class DataParserTests(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = Parser('../resources/SimLex-999.csv', '../resources/wordsim353.csv', '../resources/ultra_light_corpus.txt')
        
    def test_db_parsing(self):
#         self.assertEqual(len(parser.wordsim_db), 351)
#         self.assertEqual(len(parser.simlex_db), 999)
        self.assertEqual(self.parser.wordsim_db[frozenset({'tiger', 'cat'})].score, 7.35)
    
#     def test_corpus_parsing(self):

if __name__ == "__main__":
    unittest.main()