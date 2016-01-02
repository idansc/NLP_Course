import numpy as np

from features import *

class FeaturesManager(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, parser, feature_threshold):
        self.feature_templates = [BaseLFT1(), BaseLFT2(), BaseLFT3(), BaseLFT4(), BaseLFT5(), BaseLFT6(), BaseLFT8(), BaseLFT10(), BaseLFT13()]
        for sentence in parser.get_train_sentences():
            for labeled_token in sentence[1:]:
                head = labeled_token.head
                modifier = labeled_token.idx
                for template in self.feature_templates:
                    template.add_local_feature(sentence, head, modifier)
        
        idx = 0
        for t in self.feature_templates:
            idx = t.filter(feature_threshold, idx)
        
        self.num_features = idx       

    def calc_feature_vec(self, sentence, head, modifier):
        f = np.zeros(self.num_features)
        indices = [t.get_feature_index(sentence, head, modifier) for t in self.feature_templates if t.eval(sentence, head, modifier) == 1]
        for idx in indices:
            f[idx] += 1
            
        return f
    
    def calc_feature_vec_for_tree(self, sentence, dep_parse_tree):
        summands = [self.calc_feature_vec(sentence, head, modifier) for (head, modifier) in dep_parse_tree]
        return [sum(g) for g in summands]
    
    def get_num_features(self):
        return self.num_features
