import os
import sys
import numpy as np
import pickle
import time

from optimizer import Optimizer
from dataparser import Parser
from features_manager import FeaturesManager
from statistics import Statistics

os.chdir(os.path.dirname(__file__))

def load_parameters_vector(path):
    with open(path,'rb') as f:
        v = pickle.load(f)
    return v

def store_parameters_vector(v, path):
    with open(path, 'wb') as f:
        pickle.dump(v, f)

def learn_parameters_vector(parser, feat_manager, lambda_param, maxiter):
    print("Optimizing...")
    start_time = start_time.process_time() 
    optimizer = Optimizer(parser.get_sentences(), parser.get_num_words(), feat_manager,
                          lambda_param, maxiter)
    v = optimizer.optimize(v0=np.zeros(feat_manager.get_num_features()))
    print("Done. Elapsed time:", start_time.process_time() - start_time, "\n")
    store_parameters_vector(v, 'param_vec.dump')
    
    return v  
    
def get_param_vector(config, parser, feat_manager):
    if config['param_vector_mode'] == 'stub':
        v = np.zeros(feat_manager.get_num_features())
    elif config['param_vector_mode'] == 'learn':
        v = learn_parameters_vector(parser, feat_manager,
                                    config['learning_config']['lambda_param'],
                                    config['learning_config']['maxiter'])
    elif config['param_vector_mode'] == 'load':
        if int(os.path.dirname(config['param_vector_dump_path'])[-1]) != config['feat_threshold']:
            raise Exception("Features' threshold of dump file and Features' threshold of features extracted from training data must agree")
        v = load_parameters_vector(config['param_vector_dump_path'])
    else:
        raise Exception("Unknown param_vector_mode")
    
    return v
        
    
if __name__ == '__main__':
    config = {
        'training_data': "../resources/train.wtag",
        'test_data': "../resources/test.wtag",
        'feat_threshold': 4,
        'viterbi_tags_treshold': 6,
        'use_common_tags': False,
        'param_vector_mode': 'stub', # Options: 'stub', 'learn' or 'load'
        'learning_config': {
                'lambda_param': 50.0,
                'maxiter': 10
            },
        'param_vector_dump_path': '../resources/param_vector_dumps/baseline/iter11_threshold4/param_vec.dump'
    }
    
    print("Beginning parsing...")
    start_time = time.process_time()
    parser = Parser(config['training_data'], config['test_data'],
                    config['viterbi_tags_treshold'], config['use_common_tags'])
    print("Done. Elapsed time:", time.process_time() - start_time, "\n")
    
    print("Generating features...")
    start_time = time.process_time()
    feat_manager = FeaturesManager(parser.get_sentences(), config['feat_threshold'])
    print("Done. Elapsed time:", time.process_time() - start_time)
    
    num_features = feat_manager.get_num_features()
    if num_features == 0:
        print("No features generated. Program has been terminated.")
        sys.exit()       
    print("Number of features:", feat_manager.get_num_features(), "\n")
    
    v = get_param_vector(config, parser, feat_manager)
    print(v)
    
    stats = Statistics(parser, v, feat_manager)
    stats.print_statistics()
    
    print("Done")
