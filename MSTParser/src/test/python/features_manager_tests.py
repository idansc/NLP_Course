import unittest
import numpy as np

from dataparser import Parser
from features_manager import FeaturesManager

class FeaturesManagerTests(unittest.TestCase):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.parser = Parser("../resources/train.wtag", "../resources/test_sample.wtag", 6, False)
        
    def test_num_of_features(self):
        parser = Parser("../resources/train_simple.labeled")
        manager = FeaturesManager(parser, 1)
#         print(manager.get_num_features())
        self.assertEqual(manager.get_num_features(),
            sum(len(t.features) for t in manager.feature_templates))
        
        manager = FeaturesManager(parser, 2)
#         print(manager.get_num_features())
        self.assertEqual(manager.get_num_features(),
            sum(len(t.features) for t in manager.feature_templates))
    
    def test_calc_feature_vec(self):
        parser = Parser("../resources/train_simple.labeled")
        manager = FeaturesManager(parser, 1)
        x = parser.parse_foramtted_data("../resources/test_simple1.unlabeled")[0]
        y = {(0,2), (2,1), (2,4), (4,3)}
        self.assertTrue(
                (
                    manager.calc_feature_vec_for_tree(x, y)
                        == np.full(len(y), len(manager.feature_templates), dtype=np.int)
                ).all()
            )
        
        x = parser.parse_foramtted_data("../resources/test_simple2.unlabeled")[0]
        self.assertFalse(
            (
                manager.calc_feature_vec_for_tree(x, y)
                    == np.full(len(y), len(manager.feature_templates), dtype=np.int)
            ).all()
        )

if __name__ == "__main__":
    unittest.main()