import os
import numpy as np

from optimizer import Optimizer
from dataparser import Parser
from features_generator import FeaturesGenerator

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    
    parser = Parser("../resources/sample.wtag")
#     parser = Parser("../resources/train.wtag")
    generator = FeaturesGenerator(parser.get_word_tag_array())
    optimizer = Optimizer(parser, generator)
    v = optimizer.optimize(np.zeros(generator.get_num_features()))
    
    print(v)
