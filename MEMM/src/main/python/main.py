import os

from optimizer import Optimizer
from dataparser import Parser
from features.features_manager import FeaturesManager

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
#     opt = Optimizer()
#     opt.optimize()
    
    parser = Parser("../resources/sample.wtag")
    f_manager = FeaturesManager()
#     parser = Parser("../resources/train.wtag")
#     print(parser.get_all_tags())

