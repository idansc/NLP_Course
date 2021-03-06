import time
from utils import *
import numpy as np
from inferrer import Inferrer
from edmonds import edmonds

average_weight_vector = None

class Optimizer(object):
    '''
    Optimizes the weight vector by using the Perceptron algorithm.
    '''

    @staticmethod
    def perceptron(parser, features_manager, num_iter, save_spot):
        train_sentences = parser.get_train_sentences()
        w = np.zeros(features_manager.get_num_features())
        k = 0
        weight_vectors = []
        
        start_time = time.time()
        for i in range(num_iter):
            start_time = time.time()
            for sentence in train_sentences:
                correct = Optimizer.get_parse_tree(sentence)
                learned = Optimizer.find_arg_max(sentence, features_manager, w)
                
                if correct != learned:
                    correct_indices = features_manager.calc_feature_vec_for_tree(sentence, correct)
                    for idx, cnt in correct_indices.items():
                        w[idx] += cnt
                    
                    learned_indices = features_manager.calc_feature_vec_for_tree(sentence, learned)
                    for idx, cnt in learned_indices.items():
                        w[idx] -= cnt
                    
                    k += 1

            print("Itreration no.", i+1, "done. Elapsed time:", calc_elpased_time(start_time))
            if i in save_spot:
                weight_vectors.append(np.array(w, copy=True))
#                 path = "weight%d.dump" % i
#                 store_weight_vector(w,path)
#                 print("stored",path)
        
        if len(weight_vectors) > 0:
            global average_weight_vector
            average_weight_vector = sum(weight_vectors)/len(weight_vectors)
            print("Average weight vector was calculated:", sum(weight_vectors)/len(weight_vectors))
            
        return (w, k)
    
    @staticmethod
    def find_arg_max(sentence, features_manager, w):
        G = Inferrer.build_graph(sentence, features_manager, w, False)
        root = 0
        mst = edmonds.mst(root, G)
        
        res = set()
        for key, value in mst.items():
            for inner_key in value.keys():
                res.add((key, inner_key))
        
        return res 
            
    @staticmethod
    def get_parse_tree(sentence):
        parse_tree = set()
        for labeled_token in sentence[1:]:
            parse_tree.add((labeled_token.head, labeled_token.idx))
        
        return parse_tree
            
        
        
