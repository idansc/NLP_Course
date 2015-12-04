import os
import sys
import numpy as np
import pickle

from optimizer import Optimizer
from dataparser import Parser
from features_generator import FeaturesGenerator

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    
#     parser = Parser("../resources/sample.wtag")
    parser = Parser("../resources/train.wtag")
    generator = FeaturesGenerator(sentences=parser.get_sentences(), feat_threshold=6)
    
    num_features = generator.get_num_features()
    if num_features == 0:
        print("No features generated. Exiting program...")
        sys.exit()       
    print("Number of features:", generator.get_num_features())
    
    optimizer = Optimizer(parser.get_sentences(), generator)
    v = optimizer.optimize(v0=np.zeros(generator.get_num_features()))
      
    with open('param_vec.dump', 'wb') as f:
        pickle.dump(v, f)
      
    print(v)
    
#     with open('param_vec.dump','rb') as f:
#         v = pickle.load(f)
#     print(v)

