import unittest
import utils

from dataparser import Parser
from features_generator import FeaturesGenerator

class FeaturesGeneratorTests(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = Parser("../resources/train.wtag")
        
    def test_num_of_features(self):
        generator = FeaturesGenerator(sentences=self.parser.get_sentences(), feat_threshold=1)
        self.assertEqual(generator.get_num_features(), 24625)
        
        generator = FeaturesGenerator(sentences=self.parser.get_sentences(), feat_threshold=2)
        self.assertEqual(generator.get_num_features(), 13311)
        
        generator = FeaturesGenerator(sentences=self.parser.get_sentences(), feat_threshold=6)
        self.assertEqual(generator.get_num_features(), 5829)
    
#     def test_get_all_tags(self):
#         self.assertEqual(self.parser.get_all_tags(), utils.TAGS)
    
if __name__ == "__main__":
    unittest.main()