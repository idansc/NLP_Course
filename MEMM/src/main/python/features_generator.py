from features import *
from history import History

class FeaturesGenerator(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, word_tag_array, feat_threshold, extended_mode = False):
        self.feature_templates = [BaseFeatureTemplate1(), BaseFeatureTemplate2(), BaseFeatureTemplate3()]
        history = History() 
        for i,(word,tag) in enumerate(word_tag_array[2:-1]):
#             if word in [utils.START_SYMBOL, utils.END_SYMBOL, utils.DOT]:
#                 continue
            
            i += 2
            tm2 = word_tag_array[i-2][1]
            tm1 = word_tag_array[i-1][1]
            wm1 = word_tag_array[i-1][0]
            wp1 = word_tag_array[i+1][0]
            history.set(tm2, tm1, wm1, word, wp1, i)
            for template in self.feature_templates:
                template.add_feature(history, tag)
        
        idx = 0
        for t in self.feature_templates:
            idx = t.filter(feat_threshold, idx)        
    
    def calc_feature_vec(self, history, tag):
        return [t.get_feature_index(history, tag) for t in self.feature_templates if t.eval(history, tag) == 1]
    
    def get_num_features(self):
        return sum(len(t.features) for t in self.feature_templates)
    