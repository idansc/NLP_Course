import os
import sys
import time
import numpy as np

import pickle

from dataparser import Parser
from features_manager import FeaturesManager

os.chdir(os.path.dirname(__file__))

def load_parameters_vector(path):
    with open(path,'rb') as f:
        v = pickle.load(f)
    return v

def store_parameters_vector(v, path):
    with open(path, 'wb') as f:
        pickle.dump(v, f)
    print("Parameters vector has been saved successfully")
    
def get_weight_vector(parser, manager, config):
    if config['param_vector_mode'] == 'stub':
        w = np.zeros(manager.get_num_features())
#     elif config['param_vector_mode'] == 'learn':
#         w = learn_parameters_vector(parser, feat_manager, config['learning_config'])
#     elif config['param_vector_mode'] == 'load':
#         path_list = config['param_vector_dump_path'].split('/')
#         model = path_list [-3]
#         if (model == 'baseline' and config['use_advanced_features'] != False) or (model == 'advanced' and config['use_advanced_features'] != True):
#             raise Exception("Features' model of dump file and Features' model of features extracted from training data must agree")
#         dirname = path_list [-2]
#         if int(dirname[-1]) != config['feat_threshold']:
#             raise Exception("Features' threshold of dump file and Features' threshold of features extracted from training data must agree")
#         w = load_parameters_vector(config['param_vector_dump_path'])
    else:
        raise Exception("Unknown param_vector_mode")
    
    return w

if __name__ == '__main__':
    config = {
        'training_data': "../resources/train_sample.labeled",
        'feature_threshold': 1,
        'param_vector_mode': 'stub', # Options: 'stub', 'learn' or 'load'
    }
    
    print("Beginning parsing...")
    start_time = time.process_time()
    parser = Parser(config['training_data'])
    print("Done. Elapsed time:", (time.process_time() - start_time) / 60, "secs\n")
    
    print("Generating features...")
    start_time = time.process_time()
    manager = FeaturesManager(parser, config['feature_threshold'])
    print("Done. Elapsed time:", time.process_time() - start_time / 60, "secs\n")
    
    num_features = manager.get_num_features()
    if num_features == 0:
        print("No features generated. Program has been terminated.")
        sys.exit()       
    print("Number of features:", num_features, "\n")
    
    w = get_weight_vector(parser, manager, config)
    print("Weight vector:\n", w, "\n")
