import unittest
import numpy as np
from features_manager import FeaturesManager
from dataparser import Parser
from optimizer import Optimizer

class OptimizerTests(unittest.TestCase):
    
    def test_perceptron(self):
        parser = Parser("../resources/test_simple3.labeled")
        manager = FeaturesManager(parser, 1)
        
        self.assertEqual(Optimizer.perceptron(parser, manager, 10)[1], 2)

        
    def test_find_arg_max(self):
        parser = Parser("../resources/train_simple.labeled")
        manager = FeaturesManager(parser, 1)
        w = np.zeros(manager.get_num_features())
        x = Parser.parse_foramtted_data("../resources/test_simple3.unlabeled")[0]
        
        self.assertSetEqual(Optimizer.find_arg_max(x, manager, w), {(0, 1), (0, 2)})

    def test_get_parse_tree(self):
        x = Parser.parse_foramtted_data("../resources/test_simple4.labeled")[0]
        self.assertSetEqual(Optimizer.get_parse_tree(x), {(3, 2), (1, 3), (5, 4), (5, 6), (0, 5), (5, 1)})
        
        x = Parser.parse_foramtted_data("../resources/test_simple3.labeled")[0]
        self.assertSetEqual(Optimizer.get_parse_tree(x), {(0, 2), (2, 1)})
        
if __name__ == "__main__":
    unittest.main()