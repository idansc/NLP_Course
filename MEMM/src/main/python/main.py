import os
import sys
import numpy as np
import pickle

from optimizer import Optimizer
from dataparser import Parser
from features_manager import FeaturesManager

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':

#     parser = Parser("../resources/sample.wtag")
    parser = Parser("../resources/train.wtag")
    manager = FeaturesManager(sentences=parser.get_sentences(), feat_threshold=5)
    
    num_features = manager.get_num_features()
    if num_features == 0:
        print("No features generated. Exiting program...")
        sys.exit()       
    print("Number of features:", manager.get_num_features())
    
    optimizer = Optimizer(parser.get_sentences(), parser.get_num_words(), manager)
    v = optimizer.optimize(v0=np.zeros(manager.get_num_features()))
       
    with open('param_vec.dump', 'wb') as f:
        pickle.dump(v, f)
       
    print(v)

    print("Done")
    
#     with open('param_vec.dump','rb') as f:
#         v = pickle.load(f)
#     print(v)

