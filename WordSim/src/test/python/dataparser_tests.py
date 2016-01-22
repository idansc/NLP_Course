import unittest
from collections import Counter
from dataparser import Parser

class DataParserTests(unittest.TestCase):
#     
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.parser = Parser('../resources/SimLex-999.csv', '../resources/wordsim353.csv', '../resources/ultra_light_corpus.txt')
        
    def test_matrices(self):
        parser = Parser('../resources/test/SimLex-999.csv', '../resources/test/wordsim353.csv', '../resources/test/test_corpus.txt')
        self.assertEqual(parser.freqL1Mat.N, 19)
        self.assertEqual(len(parser.freqL1Mat.contexts), 5)
        self.assertEqual(len(parser.freqL1Mat.words), 4)
        self.assertSetEqual(set(parser.freqL1Mat.words), {'digital', 'apricot', 'pineapple', 'information'})
        self.assertDictEqual(parser.freqL1Mat.words['apricot'], Counter(pinch=1, sugar=1))
        self.assertDictEqual(parser.freqL1Mat.words['pineapple'], Counter(pinch=1, sugar=1))
        self.assertDictEqual(parser.freqL1Mat.words['digital'], Counter(computer=2, data=1, result=1))
        self.assertDictEqual(parser.freqL1Mat.words['information'], Counter(computer=1, data=6, result=4))
        self.assertEqual(parser.freqL1Mat.contexts['computer'], 3)
        self.assertEqual(parser.freqL1Mat.contexts['data'], 7)
        self.assertEqual(parser.freqL1Mat.contexts['pinch'], 2)
        self.assertEqual(parser.freqL1Mat.contexts['result'], 5)
        self.assertEqual(parser.freqL1Mat.contexts['sugar'], 2)
        
        self.assertEqual(parser.ppmiL1Mat.N, 59)
        self.assertEqual(len(parser.ppmiL1Mat.contexts), 5)
        self.assertEqual(len(parser.ppmiL1Mat.words), 4)
        self.assertSetEqual(set(parser.ppmiL1Mat.words), {'digital', 'apricot', 'pineapple', 'information'})
        self.assertEqual(round(parser.ppmiL1Mat.words['apricot']['pinch'], 2), 0.56)
        self.assertEqual(round(parser.ppmiL1Mat.words['apricot']['sugar'], 2), 0.56)
        self.assertEqual(round(parser.ppmiL1Mat.words['pineapple']['pinch'], 2), 0.56)
        self.assertEqual(round(parser.ppmiL1Mat.words['pineapple']['sugar'], 2), 0.56)
        self.assertEqual(round(parser.ppmiL1Mat.words['digital']['computer'], 2), 0.62)
        self.assertEqual(parser.ppmiL1Mat.words['digital']['data'], 0.0)
        self.assertEqual(parser.ppmiL1Mat.words['digital']['result'], 0.0)
        self.assertEqual(parser.ppmiL1Mat.words['information']['computer'], 0.0)
        self.assertEqual(round(parser.ppmiL1Mat.words['information']['data'], 2), 0.58)
        self.assertEqual(round(parser.ppmiL1Mat.words['information']['result'], 2), 0.37)
        self.assertEqual(parser.ppmiL1Mat.contexts['computer'], 3)
        self.assertEqual(parser.ppmiL1Mat.contexts['data'], 7)
        self.assertEqual(parser.ppmiL1Mat.contexts['pinch'], 2)
        self.assertEqual(parser.ppmiL1Mat.contexts['result'], 5)
        self.assertEqual(parser.ppmiL1Mat.contexts['sugar'], 2)
        
#     def test_db_parsing(self):
# #         self.assertEqual(len(parser.wordsim_db), 351)
# #         self.assertEqual(len(parser.simlex_db), 999)
#         self.assertEqual(self.parser.wordsim_db[frozenset({'tiger', 'cat'})].score, 7.35)
    

if __name__ == "__main__":
    unittest.main()