import utils

from features import *
from history import History

class FeaturesGenerator(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, word_tag_array, extended_mode = False):
        self.feature_templates = [BaseFeatureTemplate1(), BaseFeatureTemplate2(), BaseFeatureTemplate3()]
        history = History()
        index = 0    
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
                if template.get_feature_index(history, tag) == -1:
                    template.add_feature(history, tag, index)
                    index += 1
        
        self.num_features = index
        
#         print(word_tag_array)
#         print(index)
#         for template in self.feature_templates:
#             print(template.features)
    
    def calc_feature_vec(self, history, tag):
        return [template.get_feature_index(history, tag) for template in self.feature_templates if template.eval(history, tag) == 1]
    
    def get_num_features(self):
        return self.num_features
    