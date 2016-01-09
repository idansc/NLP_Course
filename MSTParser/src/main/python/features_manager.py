import numpy as np

from features import *

class FeaturesManager(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, parser, feature_threshold, extended_mode=False):
        
        self.feature_templates = []
        self.add_baseline_features()
        if extended_mode:
            self.add_extended_features()
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
        g = np.zeros(self.num_features)
        indices = [t.get_feature_index(sentence, head, modifier) for t in self.feature_templates if t.eval(sentence, head, modifier) == 1]
        for idx in indices:
            g[idx] += 1
            
        return g
    
    def calc_feature_vec_for_tree(self, sentence, dep_parse_tree):
        summands = [self.calc_feature_vec(sentence, head, modifier) for (head, modifier) in dep_parse_tree]
        return sum(np.array(g) for g in summands)
    
    def get_num_features(self):
        return self.num_features
    
    def add_baseline_features(self):
        self.feature_templates += [BasicLFT1(), BasicLFT2(), BasicLFT3(),
                                   BasicLFT4(), BasicLFT5(), BasicLFT6(),
                                   BasicLFT8(), BasicLFT10(), BasicLFT13()]
    
    def add_extended_features(self):
        self.feature_templates += [ExtendedLFT1(),ExtendedLFT2(),ExtendedLFT3(),
                                   ExtendedLFT4(), ExtendedLFT5(), ExtendedLFT6(),
                                   ExtendedLFT7(), ExtendedLFT8(), ExtendedLFT9()]

