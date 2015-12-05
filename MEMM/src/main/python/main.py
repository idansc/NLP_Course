import os
import sys
import numpy as np
import pickle
import time

from optimizer import Optimizer
from dataparser import Parser
from features_manager import FeaturesManager
from inference import Inference

os.chdir(os.path.dirname(__file__))

def learn_parameter_vector(lambda_param, feat_threshold, maxiter):
    t_total = time.process_time()
    
    print("begin finding Lambda:", lambda_param, "; Features threshold:", feat_threshold, "; Max iterations:", maxiter)
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
    training_data = "../resources/sample.wtag"
    test_data = "../resources/test.wtag"
    print("Beginning learning. Data:", training_data,"Parsing...")
    t_step = time.process_time()
    parser = Parser(training_data,test_data)
    print("Done. Elapsed time:", time.process_time() - t_step)
#     v = learn_parameter_vector( lambda_param=50.0, feat_threshold=4, maxiter=7)
#    v = learn_parameter_vector(training_data="../resources/train.wtag", lambda_param=50.0, feat_threshold=5, maxiter=14)
#    store_parameter_vector(v, 'param_vec.dump')
    
#     v = load_parameter_vector('../resources/param_vector_dumps/baseline/iter11_threshold4/param_vec.dump')
    inference = Inference(parser)
    for s in parser.get_test_sentences():
        res = inference.viterbi(s)
    print("Done")
