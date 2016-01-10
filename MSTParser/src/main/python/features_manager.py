import numpy as np

from features import *

class FeaturesManager(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, parser, feature_threshold, prefix_flag, extended_mode=False):

        self.parser = parser
        self.prefix_flag = prefix_flag
        self.feature_templates = []
        self.add_baseline_features()
        if extended_mode:
            self.add_extended_features()
            self.add_in_between_features()
        for sentence in parser.get_train_sentences():
            for labeled_token in sentence[1:]:
                head = labeled_token.head
                modifier = labeled_token.idx
                for template in self.feature_templates:
                    template.add_local_feature(sentence, head, modifier)
        
        idx = 0
        for i, t in enumerate(self.feature_templates):
            temp = idx
            idx = t.filter(feature_threshold, idx)
            print("template number",i,":", idx - temp)


        
        self.num_features = idx       

    def calc_feature_vec(self, sentence, head, modifier):
        return [t.get_feature_index(sentence, head, modifier) for t in self.feature_templates if t.eval(sentence, head, modifier) == 1]
    
    def calc_feature_vec_for_tree(self, sentence, dep_parse_tree):
        result = {}
        for (head, modifier) in dep_parse_tree:
            indices = self.calc_feature_vec(sentence, head, modifier)
            for idx in indices:
                cnt = result.get(idx, 0)
                result[idx] = cnt + 1
        
        return result
    
    def get_num_features(self):
        return self.num_features
    
    def add_baseline_features(self):
        self.feature_templates += [BasicLFT1(), BasicLFT2(), BasicLFT3(),
                                   BasicLFT4(), BasicLFT5(), BasicLFT6(),
                                   BasicLFT8(), BasicLFT10(), BasicLFT13()]
    
    def add_extended_features(self):
        self.feature_templates += [ExtendedLFT1(),ExtendedLFT2(),ExtendedLFT3(),
                                   ExtendedLFT4(), ExtendedLFT5(), ExtendedLFT6(),
                                   ExtendedLFT7(), ExtendedLFT8(), ExtendedLFT9()
                                  ]
        if self.prefix_flag:
             self.feature_templates += [ExtendedLFT10(self.parser.get_prefix())]

    
    def add_in_between_features(self):
        TAGS = {'CD', 'POS', 'UH', 'JJR', 'PRP', 'WDT', 'EX', 'SYM', 'NNPS',
                'WP', 'CC', 'JJ', 'VBP', 'WRB', 'RBR', 'MD', 'IN', 'VB', 'DT',
                'RBS', 'VBG', 'RP', 'JJS', 'NN', 'PDT', 'RB', 'VBN', 'TO', 'LS',
                'NNS', 'VBZ', 'FW', 'NNP', 'VBD'}
        
        self.feature_templates += [InBetweenLFT1(tag) for tag in TAGS]
        self.feature_templates += [InBetweenLFT2()]