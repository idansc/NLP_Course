import unittest

from dataparser import Parser

class DataParserTests(unittest.TestCase):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.parser = Parser("../resources/train.wtag", "../resources/test_sample.wtag", 6, False)
        
    def test_num_of_words(self):
        parser = Parser("../resources/train_sample.labeled")
        sentences = parser.get_train_sentences()
        self.assertEqual(sentences[1][12][3], "3")
        self.assertEqual(sentences[1][6][1], "N.V.")
    

if __name__ == "__main__":
    unittest.main()