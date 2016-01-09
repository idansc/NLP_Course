import os
import sys
import time
import numpy as np
import pickle
from dataparser import Parser
from features_manager import FeaturesManager
from inferrer import Inferrer
from optimizer import Optimizer

os.chdir(os.path.dirname(__file__))

def load_weight_vector(path):
    with open(path,'rb') as f:
        w = pickle.load(f)
    return w

def store_weight_vector(w, path):
    with open(path, 'wb') as f:
        pickle.dump(w, f)
    print("Parameters vector has been saved successfully")
    
def learn_weight_vector(parser, feat_manager, config):
    print("Optimizing...")
    start_time = time.process_time() 
    result = Optimizer.perceptron(parser, feat_manager, config['num_iter'])
    print("Done. K:", result[1], "; Elapsed time:", time.process_time() - start_time, "\n")
    store_weight_vector(result[0], 'weights.dump')
    
    return result[0] 
    
def get_weight_vector(parser, manager, config):
    if config['param_vector_mode'] == 'stub':
        w = np.zeros(manager.get_num_features())
    elif config['param_vector_mode'] == 'learn':
        w = learn_weight_vector(parser, manager, config['learning_config'])
    elif config['param_vector_mode'] == 'load':
        path_list = config['param_vector_dump_path'].split('/')
        model = path_list [-3]
        if (model == 'baseline' and config['extended_mode'] != False) or (model == 'extended' and config['extended_mode'] != True):
            raise Exception("Features' model of dump file and Features' model of features extracted from training data must agree")
#         dirname = path_list [-2]
#         if int(dirname[-1]) != config['feat_threshold']:
#             raise Exception("Features' threshold of dump file and Features' threshold of features extracted from training data must agree")
        w = load_weight_vector(config['param_vector_dump_path'])
    else:
        raise Exception("Unknown param_vector_mode")
    
    return w

if __name__ == '__main__':
    config = {
        'training_data': "../resources/train.labeled",
        'feature_threshold': 5,
        'extended_mode': True,
        'param_vector_mode': 'learn', # Options: 'stub', 'learn' or 'load'
        'learning_config': {'num_iter': 10},
        'param_vector_dump_path': "../resources/weight_vector_dumps/baseline/w01/weights.dump",
        'input_data': "../resources/test.labeled",
#         'input_data': "../resources/comp.unlabeled",
        'output_path': "results.labeled"

    }
    
    print("Beginning parsing...")
    start_time = time.process_time()
    parser = Parser(config['training_data'])
    print("Done. Elapsed time:", (time.process_time() - start_time) / 60, "secs\n")
    
    print("Generating features...")
    start_time = time.process_time()
    manager = FeaturesManager(parser, config['feature_threshold'], config['extended_mode'])
    print("Done. Elapsed time:", time.process_time() - start_time / 60, "secs\n")
    
    num_features = manager.get_num_features()
    if num_features == 0:
        print("No features generated. Program has been terminated.")
        sys.exit()       
    print("Number of features:", num_features, "\n")
    
    w = get_weight_vector(parser, manager, config)
    print("Weight vector: (", config['param_vector_mode'], ")\n", w, "\n")
    
    print("Inferring...")
    start_time = time.process_time()
    inferrer = Inferrer(config['input_data'], parser, manager, w)
    print("Done. Elapsed time:", time.process_time() - start_time / 60, "secs\n")
    
    inferrer.print_statistics()
#     inferrer.store(config['output_path'])
