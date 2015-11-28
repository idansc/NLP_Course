import sys

from utils import START_SYMBOL, END_SYMBOL, EOF_SYMBOL

class Parser(object):
    '''
    Handles parsing and related aspects
    '''
    
    START_TUPLE = (START_SYMBOL, START_SYMBOL)
    END_TUPLE = (END_SYMBOL, END_SYMBOL)
    EOF_TUPLE = (EOF_SYMBOL, EOF_SYMBOL)
    
    def __init__(self, filepath):
        self.word_tag_array = []
        self.parse(filepath)

    def parse(self,filepath):
        with open(filepath, 'r') as datafile:
            for line in datafile:
                line = line.strip()
                parsed_phrased = \
                    [self.START_TUPLE, self.START_TUPLE] \
                    + [tuple(w.split('_')) for w in line.split(' ')] \
                    + [self.END_TUPLE]
                
                self.word_tag_array += parsed_phrased
            
            self.word_tag_array += [self.EOF_TUPLE]
        
        if len(self.word_tag_array) == 0:
            print("Training data is empty. Exiting program...")
            sys.exit()
    
    def get_word_tag_array(self):
        return self.word_tag_array
    
    def get_all_tags(self):
        return {wt[1] for wt in self.word_tag_array}
    
    def get_num_words(self):
        return len(self.word_tag_array)
    
        