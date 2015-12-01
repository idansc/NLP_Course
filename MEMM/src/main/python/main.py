import os
import numpy as np
import pickle

from optimizer import Optimizer
from dataparser import Parser
from features_generator import FeaturesGenerator

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    
#     parser = Parser("../resources/sample.wtag")
    parser = Parser("../resources/train.wtag")
    generator = FeaturesGenerator(word_tag_array=parser.get_word_tag_array(), feat_threshold=5)
    print("Number of features:", generator.get_num_features())
    optimizer = Optimizer(parser, generator)
    v = optimizer.optimize(v0=np.zeros(generator.get_num_features()))
#     v = optimizer.optimize(np.ones(generator.get_num_features()))
     
    with open('param_vec.dump', 'wb') as f:
        pickle.dump(v, f)
     
    print(v)
    
#     with open('param_vec.dump','rb') as f:
#         v = pickle.load(f)
#     print(v)

