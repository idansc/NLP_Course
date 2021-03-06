import unittest
import constants
import numpy as np
import utils

from math import exp
from dataparser import Parser
from features_manager import FeaturesManager
from history import History

class FeaturesManagerTests(unittest.TestCase):
     
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#     def test_num_of_features(self):
#         parser = Parser("../resources/train.wtag", "../resources/test_sample.wtag", 6, False)
#         
#         manager = FeaturesManager(parser, feat_threshold=1)
#         self.assertEqual(manager.get_num_features(), 24625)
#         
#         manager = FeaturesManager(parser, feat_threshold=2)
#         self.assertEqual(manager.get_num_features(), 13311)
#         
#         manager = FeaturesManager(parser, feat_threshold=6)
#         self.assertEqual(manager.get_num_features(), 5829)
    
    def test_num_features(self):
        parser = Parser("../resources/features_test/test_num_featurese.wtag", "../resources/test_sample.wtag", 6, False)
        feat_manager = FeaturesManager(parser, feat_threshold=1, use_advanced_features=False)
        self.assertEqual(feat_manager.get_num_features(), 17)
        
        feat_manager = FeaturesManager(parser, feat_threshold=2, use_advanced_features=False)
        self.assertEqual(feat_manager.get_num_features(), 1)
    
    def test_calc_feature_vec(self):
        parser = Parser("../resources/features_test/test_calc_feature_vec.wtag", "../resources/test_sample.wtag", 6, False)
        feat_manager = FeaturesManager(parser, feat_threshold=1, use_advanced_features=False)
        self.assertEqual(feat_manager.get_num_features(), 9)
        
        history = History()
        tm2 = "DT"
        tm1 = "NN"
        w = "concern"
        wm1 = "biotechnology"
        wp1 = constants.END_SYMBOL
        history.set(tm2, tm1, wm1, w, wp1)
        tag = "NN"
        self.assertEqual(feat_manager.calc_feature_vec(history, tag), [2, 5, 8])
        
    def test_calc_prob(self):
        history = History()
        tm2 = constants.START_SYMBOL
        tm1 = "STUB"
        w = "STUB"
        wm1 = "STUB"
        wp1 = "STUB"
        history.set(tm2, tm1, wm1, w, wp1)
        tag = "STUB"
        
        parser = Parser("../resources/features_test/test_calc_prob.wtag", "../resources/test_sample.wtag", 6, False)
        feat_manager = FeaturesManager(parser, feat_threshold=1, use_advanced_features=False)
        
        v = np.zeros(feat_manager.get_num_features())
        self.assertEqual(utils.calc_prob(feat_manager, v, history, tag), 1 / len(constants.TAGS))
        
        tm2 = constants.START_SYMBOL
        tm1 = "DT"
        w = "biotechnology"
        wm1 = "The"
        wp1 = "concern"
        history.set(tm2, tm1, wm1, w, wp1)
        tag = "NN"
        self.assertEqual(utils.calc_prob(feat_manager, v, history, tag), 1 / len(constants.TAGS))

        v = np.ones(feat_manager.get_num_features())
        self.assertEqual(utils.calc_prob(feat_manager, v, history, tag), exp(3) / (len(constants.TAGS) - 1.0 + exp(3)))
    
    def test_suffixes_features(self):
        parser = Parser("../resources/features_test/test_suffixes_features.wtag", "../resources/test_sample.wtag", 6, False)
        feat_manager = FeaturesManager(parser, feat_threshold=2, use_advanced_features=True)
        self.assertEqual(feat_manager.get_num_features(), 2)

    def test_prefixes_features(self):
        parser = Parser("../resources/features_test/test_prefixes_features.wtag", "../resources/test_sample.wtag", 6, False)
        feat_manager = FeaturesManager(parser, feat_threshold=2, use_advanced_features=True)
        self.assertEqual(feat_manager.get_num_features(), 2)

if __name__ == "__main__":
    unittest.main()