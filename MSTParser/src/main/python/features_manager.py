from features import *

class FeaturesManager(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, parser, feature_threshold):
        self.feature_templates = [BaseLFT1(), BaseLFT2()]
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

    def calc_feature_vec(self, sentence, dep_parse_tree):
        local_feature_vectors = []
        for (head, modifier) in dep_parse_tree:
            local_feature_vectors.append(
                    [t.eval(sentence, head, modifier) for t in self.feature_templates]
                )

        return [sum(g) for g in local_feature_vectors] 
    
    def get_num_features(self):
        return self.num_features
