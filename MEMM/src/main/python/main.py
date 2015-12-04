import os
import sys
import numpy as np
import pickle
import time

from optimizer import Optimizer
from dataparser import Parser
from features_manager import FeaturesManager

os.chdir(os.path.dirname(__file__))

def learn_parameter_vector(training_data, lambda_param, feat_threshold, maxiter):
    t_total = time.process_time()
    
    print("Beginning learning. Data:", training_data, "; Lambda:", lambda_param, "; Features threshold:", feat_threshold, "; Max iterations:", maxiter)
    print()
    print("Parsing...")
    t_step = time.process_time()
    parser = Parser(training_data)
    print("Done. Elapsed time:", time.process_time() - t_step)
    print()
    print("Generating features...")
    t_step = time.process_time()
    manager = FeaturesManager(parser.get_sentences(), feat_threshold)
    print("Done. Elapsed time:", time.process_time() - t_step)
     
    num_features = manager.get_num_features()
    if num_features == 0:
        print("No features generated. Exiting program...")
        sys.exit()       
    print("Number of features:", manager.get_num_features())
    print()
    print("Optimizing...")
    t_step = time.process_time() 
    optimizer = Optimizer(parser.get_sentences(), parser.get_num_words(), manager, lambda_param, maxiter)
    v = optimizer.optimize(v0=np.zeros(manager.get_num_features()))
    print("Optimization is done. Elapsed time:", time.process_time() - t_step)
    print()
    print("Learning is done. Total elapsed time:", time.process_time() - t_total)
    return v

def load_parameter_vector(path):
    with open(path,'rb') as f:
        v = pickle.load(f)
    return v

def store_parameter_vector(v, path):
    with open(path, 'wb') as f:
        pickle.dump(v, f)
    
if __name__ == '__main__':
#     v = learn_parameter_vector(training_data="../resources/sample.wtag", lambda_param=50.0, feat_threshold=4, maxiter=7)
    v = learn_parameter_vector(training_data="../resources/train.wtag", lambda_param=50.0, feat_threshold=5, maxiter=14)
    store_parameter_vector(v, 'param_vec.dump')
    
#     v = load_parameter_vector('../resources/param_vector_dumps/baseline/iter11_threshold4/param_vec.dump')
    print("Done")
