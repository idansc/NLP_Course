import unittest
import numpy as np
import constants
import utils

from dataparser import Parser
from features_manager import FeaturesManager
from optimizer import Optimizer
from history import History

class OptimizerTests(unittest.TestCase):
     
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
    def test_param_vector(self):
        parser = Parser("../resources/optimizer_test/test_param_vector.wtag", "../resources/test_sample.wtag", 6, False)
        feat_manager = FeaturesManager(parser, feat_threshold=1)
        self.assertEqual(feat_manager.get_num_features(), 17)
        
        optimizer = Optimizer(parser, parser.get_num_words(), feat_manager,
                          lambda_param=50.0, maxiter=50)
        v = optimizer.optimize(v0=np.zeros(feat_manager.get_num_features()))
        print(v)
        
        history = History()
        tm2 = constants.START_SYMBOL
        tm1 = "DT"
        w = "biotechnology"
        wm1 = "The"
        wp1 = "concern"
        history.set(tm2, tm1, wm1, w, wp1)
        tag = "NN"
        
        self.assertGreaterEqual(
                utils.calc_prob(feat_manager, v, history, tag),
                utils.calc_prob(feat_manager, np.zeros(feat_manager.get_num_features()), history, tag)
            )

if __name__ == "__main__":
    unittest.main()