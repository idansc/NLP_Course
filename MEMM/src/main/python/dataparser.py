import sys

from utils import START_SYMBOL, END_SYMBOL

class Parser(object):
    '''
    Handles parsing and related aspects
    '''
    
    START_TUPLE = (START_SYMBOL, START_SYMBOL)
    END_TUPLE = (END_SYMBOL, END_SYMBOL)
    
    def __init__(self, filepath):
        self.sentences = []
        self.parse(filepath)

    def parse(self,filepath):
        with open(filepath, 'r') as datafile:
            for line in datafile:
                line = line.strip()
                sentence = \
                    [self.START_TUPLE, self.START_TUPLE] \
                    + [tuple(w.split('_')) for w in line.split(' ')] \
                    + [self.END_TUPLE]
                
                self.sentences.append(sentence)
        
        if len(self.sentences) == 0:
            print("Training data is empty. Exiting program...")
            sys.exit()
    
    def get_sentences(self):
        return self.sentences
    
    def get_all_tags(self):
        tags = set()
        for s in self.sentences:
            for wt in s:
                tags.add(wt[1])
        
        return tags
    
    def get_num_words(self):
        return sum(len(s) - 3 for s in self.sentences)
    
        