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
    print("Parameters vector has been saved successfully")

def learn_parameters_vector(parser, feat_manager, config):
    print("Optimizing...")
    start_time = time.process_time() 
    optimizer = Optimizer(parser, feat_manager,
                          config['lambda_param'], config['maxiter'], config['specific_words_tag'])
    v = optimizer.optimize(v0=np.zeros(feat_manager.get_num_features()))
    print("Done. Elapsed time:", time.process_time() - start_time, "\n")
    store_parameters_vector(v, 'param_vec.dump')
    
    return v  
    
def get_param_vector(config, parser, feat_manager):
    if config['param_vector_mode'] == 'stub':
        v = np.zeros(feat_manager.get_num_features())
    elif config['param_vector_mode'] == 'learn':
        v = learn_parameters_vector(parser, feat_manager, config['learning_config'])
    elif config['param_vector_mode'] == 'load':
        path_list = config['param_vector_dump_path'].split('/')
        model = path_list [-3]
        if (model == 'baseline' and config['use_advanced_features'] != False) or (model == 'advanced' and config['use_advanced_features'] != True):
            raise Exception("Features' model of dump file and Features' model of features extracted from training data must agree")
        dirname = path_list [-2]
        if int(dirname[-1]) != config['feat_threshold']:
            raise Exception("Features' threshold of dump file and Features' threshold of features extracted from training data must agree")
        v = load_parameters_vector(config['param_vector_dump_path'])
    else:
        raise Exception("Unknown param_vector_mode")
    
    return v
        
    
if __name__ == '__main__':
    config = {
        'training_data': "../resources/train.wtag",
        'test_data': "../resources/test.wtag",
        'comp_data': "../resources/comp.words",
        'feat_threshold': 5,
        'viterbi_tags_treshold': 30,
        'use_advanced_features': False,
        'use_common_tags': False,
        'param_vector_mode': 'load', # Options: 'stub', 'learn' or 'load'
        'learning_config': {
                'lambda_param': 70.0,
                'maxiter': 16,
                'specific_words_tag': False
            },
        'param_vector_dump_path': '../resources/param_vector_dumps/baseline/iter16_threshold5/param_vec.dump',
        'specific_words_tag': True,
        'confusion_matrix_print': False,
        'competition': True
    }
    
    print("Beginning parsing...")
    start_time = time.process_time()
    parser = Parser(config['training_data'], config['test_data'], config['comp_data'],
                    config['viterbi_tags_treshold'], config['use_common_tags'])
    print("Done. Elapsed time:", time.process_time() - start_time, "\n")
    
    print("Generating features...")
    start_time = time.process_time()
    feat_manager = FeaturesManager(parser, config['feat_threshold'], config['use_advanced_features'])
    print("Done. Elapsed time:", time.process_time() - start_time)
    
    num_features = feat_manager.get_num_features()
    if num_features == 0:
        print("No features generated. Program has been terminated.")
        sys.exit()       
    print("Number of features:", feat_manager.get_num_features(), "\n")
    
    v = get_param_vector(config, parser, feat_manager)
    print(v, "\n")
    
    print("Extracting results from test data...")
    start_time = time.process_time()
    stats = Statistics(parser, v, feat_manager,config)
    stats.print_statistics()    
    print("Done extracting results from test data. Elapsed time:", time.process_time() - start_time)
    print("Done. Elapsed time:", time.process_time() - start_time)



