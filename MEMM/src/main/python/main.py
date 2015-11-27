import os

from optimizer import Optimizer
from dataparser import Parser

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
#     opt = Optimizer()
#     opt.optimize()
    
    parser = Parser("../resources/sample.wtag")

#     print(get_parsed_data("../resources/train.wtag"))




