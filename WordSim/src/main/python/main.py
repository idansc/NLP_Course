import os
from utils import *
from dataparser import Parser
from evaluator import Evaluator

os.chdir(os.path.dirname(__file__))

def run(config):
    print("Beginning parsing...")
    start_time = time.time()
    parser = Parser(config['simlex_path'], config['wordsim_path'], config['corpus'])
    print("Done parsing. Elapsed time:", calc_elpased_time(start_time), "\n")
    
    print("Beginning evaluation...")
    start_time = time.time()
    evaluator_cosine = Evaluator(parser)
    evaluator_cosine.evaluate()

def build_comp_files():
    config = {
        'simlex_path': 'SimLex-999.csv',
        'wordsim_path': 'wordsim353.csv',
        'corpus': 'full.txt'
    }
    run(config)

if __name__ == '__main__':
    config = {
        'simlex_path': '../resources/SimLex-999.csv',
        'wordsim_path': '../resources/wordsim353.csv',
        'corpus': '../resources/full.txt'
#         'corpus': '../resources/light_corpus.txt'
#         'corpus': '../resources/ultra_light_corpus.txt'
    }
    run(config)   

