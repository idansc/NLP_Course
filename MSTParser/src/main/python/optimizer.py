import numpy as np
from inferrer import Inferrer
from edmonds import edmonds

class Optimizer(object):
    '''
    Optimizes the weight vector by using the Perceptron algorithm.
    '''

    @staticmethod
    def perceptron(parser, features_manager, num_iter):
        train_sentences = parser.get_train_sentences()
        w = np.zeros(features_manager.get_num_features())
        k = 0
        
        for _ in range(num_iter):
            for sentence in train_sentences:
                correct = Optimizer.get_parse_tree(sentence)
                learned = Optimizer.find_arg_max(sentence, features_manager, w)
                
                if correct != learned:
                    w = w + features_manager.calc_feature_vec_for_tree(sentence, correct) - features_manager.calc_feature_vec_for_tree(sentence, learned)
                    k += 1
            
        return (w,k)
    
    @staticmethod
    def find_arg_max(sentence, features_manager, w):
        G = Inferrer.build_graph(sentence, features_manager, w)
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
            
        
        
        