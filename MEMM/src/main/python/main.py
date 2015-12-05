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

def load_parameters_vector(path):
    with open(path,'rb') as f:
        v = pickle.load(f)
    return v

def store_parameters_vector(v, path):
    with open(path, 'wb') as f:
        pickle.dump(v, f)

def learn_parameters_vector(parser, feat_manager, lambda_param, maxiter):
    print("Optimizing...")
    t_step = time.process_time() 
    optimizer = Optimizer(parser.get_sentences(), parser.get_num_words(), feat_manager,
                          lambda_param, maxiter)
    v = optimizer.optimize(v0=np.zeros(feat_manager.get_num_features()))
    print("Done. Elapsed time:", time.process_time() - t_step, "\n")
    store_parameters_vector(v, 'param_vec.dump')
    
    return v  
    
if __name__ == '__main__':
    
#    training_data = "../resources/training_sample.wtag"
    training_data = "../resources/train.wtag"
    test_data = "../resources/test.wtag"
    
    print("Beginning parsing...")
    t_step = time.process_time()
    parser = Parser(training_data, test_data, viterbi_tags_treshold=6, use_common_tags=False)
    print("Done. Elapsed time:", time.process_time() - t_step, "\n")
    
    print("Generating features...")
    t_step = time.process_time()
    feat_manager = FeaturesManager(parser.get_sentences(), feat_threshold=4)
    print("Done. Elapsed time:", time.process_time() - t_step)
    
    num_features = feat_manager.get_num_features()
    if num_features == 0:
        print("No features generated. Exiting program...")
        sys.exit()       
    print("Number of features:", feat_manager.get_num_features())
    print()
    
    # *** Loading or learning parameters vector *** 
    
#    v = learn_parameters_vector(parser, feat_manager, lambda_param=50.0, maxiter=10)
    v = load_parameters_vector('../resources/param_vector_dumps/baseline/iter11_threshold4/param_vec.dump')
    print(v)
    
    inference = Inference(parser, v, feat_manager)
    for s in parser.get_test_sentences():
        res = inference.viterbi(s)
        print(res)
    print("Done")
