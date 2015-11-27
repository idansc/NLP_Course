from utils import START_SYMBOL
from utils import END_SYMBOL

class Parser(object):
    '''
    A centralized class that handles the parsing aspects
    '''
    
    START_TUPLE = (START_SYMBOL,START_SYMBOL)
    END_TUPLE = (END_SYMBOL,END_SYMBOL)
    
    def __init__(self, filepath):
        self.word_tag_array = []
        self.parse(filepath)

    def parse(self,filepath):
        with open(filepath, 'r') as datafile:
            for line in datafile:
                parsed_phrased = \
                    [self.START_TUPLE, self.START_TUPLE] \
                    + [tuple(w.split('_')) for w in line.split(' ')] \
                    + [self.END_TUPLE]
                
                self.word_tag_array += parsed_phrased
    
        