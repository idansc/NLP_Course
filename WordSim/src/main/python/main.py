import os
import time
from utils import *
from dataparser import Parser

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    config = {
        'simlex_path': '../resources/SimLex-999.csv',
        'wordsim_path': '../resources/wordsim353.csv',
#         'corpus': '../resources/full.csv.txt',
        'corpus': '../resources/light_corpus.txt'
    }
    
    # Preprocessing
        # parse SimLex and WordSim
        # parse trainging data
        #  
    
    print("Beginning parsing...")
    start_time = time.time()
    parser = Parser(config['simlex_path'], config['wordsim_path'], config['corpus'])
    print("Done. Elapsed time:", calc_elpased_time(start_time), "\n")
    
#     
