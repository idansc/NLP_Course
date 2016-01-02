import unittest
import numpy as np

from dataparser import Parser
from features_manager import FeaturesManager
from inferrer import Inferrer

class InferrerTests(unittest.TestCase):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.parser = Parser("../resources/train.wtag", "../resources/test_sample.wtag", 6, False)
    
    def test_build_graph(self):
        parser = Parser("../resources/train_simple.labeled")
        manager = FeaturesManager(parser, 1)
        w = np.zeros(manager.get_num_features())
        x = parser.parse_foramtted_data("../resources/test_simple3.unlabeled")[0]
        
        self.assertDictEqual(Inferrer.build_graph(x, manager, w), {0: {1: -0.0, 2: -0.0}, 1: {2: -0.0}, 2: {1: -0.0}})
    
    def test_reformat_dep_tree(self):
        G = {
                0: {1: 2},
        }
        self.assertDictEqual(Inferrer.reformat_dep_tree(G), {1: 0})
        
        G = {
                0: {3: 2},
                1: {2: 1},
                3: {1: 3, 4: 2}
        }
        self.assertDictEqual(Inferrer.reformat_dep_tree(G), {1: 3, 2: 1, 3: 0, 4: 3})
    
    
    def test_print_statistics(self):
        parser = Parser("../resources/train_simple.labeled")
        manager = FeaturesManager(parser, 1)
        w = np.zeros(manager.get_num_features())
#         
        inferrer = Inferrer("../resources/test_simple3.unlabeled", parser, manager, w)
        
        with self.assertRaises(Exception):
            inferrer.print_statistics()
    
    
    def test_store(self):
        parser = Parser("../resources/train_simple.labeled")
        manager = FeaturesManager(parser, 1)
        w = np.zeros(manager.get_num_features())
#         
        inferrer = Inferrer("../resources/test_simple3.unlabeled", parser, manager, w)
        
        outpath = "../resources/test_simple3.res"
        inferrer.store(outpath)
        
        with open(outpath, 'r') as outfile:
            data = outfile.read()
        
        self.assertEqual(data, "1\tJohn\t_\tNNP\t_\t_\t0\t_\t_\t_\n2\tsaw\t_\tVB\t_\t_\t0\t_\t_\t_\n\n")
        
if __name__ == "__main__":
    unittest.main()