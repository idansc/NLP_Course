from features import *
from history import History

class FeaturesGenerator(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, word_tag_array, extended_mode = False):
        self.features = [BaseFeature1(), BaseFeature2(), BaseFeature3()]
        history = History()      
        for i,(word,tag) in enumerate(word_tag_array[2:-1]):
            i += 2
            tm2 = word_tag_array[i-2][1]
            tm1 = word_tag_array[i-1][1]
            wm1 = word_tag_array[i-1][0]
            wp1 = word_tag_array[i+1][0]
            history.set(tm2, tm1, wm1, word, wp1, i)
            for feature in self.features:
                feature.add_sample(history, tag)
        
#         for f in self.features:
#             print(f.samples)
    
    def get_features(self):
        return self.features