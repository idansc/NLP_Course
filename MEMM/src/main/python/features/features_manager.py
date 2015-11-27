# from scipy.sparse import lil_matrix

from features.features_set import *
from history import History

class FeaturesManager(object):
    '''
    Constructs set of features out of a given training data
    '''

    def __init__(self, extended_mode = False):
        self.extended_mode = extended_mode
        self.features = [BaseFeature1(), BaseFeature2(), BaseFeature3()]
    
    def train(self, word_tag_array):
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
    
#     def foo(self):
#         n = len(word_tag_array)
#         m = n * (3 if self.extended_mode is True else 8)
#         F = lil_matrix(n, m)
#         print(n,m)
#         print(F)
#         pass
        