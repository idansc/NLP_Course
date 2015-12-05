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

def load_parameter_vector(path):
    with open(path,'rb') as f:
        v = pickle.load(f)
    return v

def store_parameter_vector(v, path):
    with open(path, 'wb') as f:
        pickle.dump(v, f)
    
if __name__ == '__main__':
    
#     training_data = "../resources/training_sample.wtag"
    training_data = "../resources/train.wtag"
    test_data = "../resources/test_sample.wtag"
    
    print("Beginning parsing...")
    t_step = time.process_time()
    parser = Parser(training_data, test_data, viterbi_tags_treshold=6, use_common_tags=False)
    print("Done. Elapsed time:", time.process_time() - t_step, "\n")
    
    print("Generating features...")
    t_step = time.process_time()
    feat_manager = FeaturesManager(parser.get_sentences(), feat_threshold=6)
    print("Done. Elapsed time:", time.process_time() - t_step)
    
    num_features = feat_manager.get_num_features()
    if num_features == 0:
        print("No features generated. Exiting program...")
        sys.exit()       
    print("Number of features:", feat_manager.get_num_features())
    print()
    
    # *** Loading or learning parameters vector *** 
    
#     print("Optimizing...")
#     t_step = time.process_time() 
#     optimizer = Optimizer(parser.get_sentences(), parser.get_num_words(), feat_manager,
#                            lambda_param=50.0, maxiter = 10)
#     v = optimizer.optimize(v0=np.zeros(num_features.get_num_features()))
#     print("Done. Elapsed time:", time.process_time() - t_step)
#     store_parameter_vector(v, 'param_vec.dump')
    
#     v = load_parameter_vector('../resources/param_vector_dumps/baseline/iter11_threshold4/param_vec.dump')
    
    # Stub parameters vector
    v = np.zeros(5000)
    
    inference = Inference(parser, v, feat_manager)
    for s in parser.get_test_sentences():
        res = inference.viterbi(s)
        print(res)
    
    print("Done")
