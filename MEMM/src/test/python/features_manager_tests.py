import unittest
import utils

from dataparser import Parser
from features_manager import FeaturesManager

class FeaturesManagerTests(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = Parser("../resources/train.wtag")
        
    def test_num_of_features(self):
        manager = FeaturesManager(sentences=self.parser.get_sentences(), feat_threshold=1)
        self.assertEqual(manager.get_num_features(), 24625)
        
        manager = FeaturesManager(sentences=self.parser.get_sentences(), feat_threshold=2)
        self.assertEqual(manager.get_num_features(), 13311)
        
        manager = FeaturesManager(sentences=self.parser.get_sentences(), feat_threshold=6)
        self.assertEqual(manager.get_num_features(), 5829)
    
if __name__ == "__main__":
    unittest.main()