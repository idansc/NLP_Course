
class FeaturesManager(object):
    '''
    Constructs set of features out of a given training data
    '''

    def __init__(self, extended_mode = False):
        self.extended_mode = extended_mode
    
    def train(self, word_tag_array):
        for i,(word,tag) in enumerate(word_tag_array):
            print(i,word,tag)
        