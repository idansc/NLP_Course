import os
from utils import *
from dataparser import Parser
from evaluator import Evaluator

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    config = {
        'simlex_path': '../resources/SimLex-999.csv',
        'wordsim_path': '../resources/wordsim353.csv',
#         'corpus': '../resources/full.txt'
        'corpus': '../resources/light_corpus.txt'
#         'corpus': '../resources/ultra_light_corpus.txt'
    }
    
    print("Beginning parsing...")
    start_time = time.time()
    parser = Parser(config['simlex_path'], config['wordsim_path'], config['corpus'])
    print("Done. Elapsed time:", calc_elpased_time(start_time), "\n")
    
    print("Beginning evaluation...")
    start_time = time.time()
    evaluator = Evaluator(parser)
    evaluator.evaluate()
    print("Done. Elapsed time:", calc_elpased_time(start_time), "\n")
    
#     
