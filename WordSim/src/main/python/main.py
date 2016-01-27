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
        'simlex_path': os.path.join("..", "resources", "SimLex-999.csv"),
        'wordsim_path': os.path.join("..", "resources", "wordsim353.csv"),
        'corpus': os.path.join("..", "resources", "light_corpus.txt")
#         'corpus': os.path.join("..", "resources", "ultra_light_corpus.txt")
    }
    run(config)   

