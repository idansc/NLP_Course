import sys

from labeled_token import LabeledToken
from constants import ROOT_SYMBOL

class Parser(object):
    '''
    Handles data parsing
    '''

    def __init__(self, training_data_path):
        self.train_sentences = self.parse_foramtted_data(training_data_path)
        
    
    @staticmethod    
    def parse_foramtted_data(filepath):
        result = []
        with open(filepath, 'r') as datafile:
            labeled_sentence = []
            for line in datafile:
                line = line.strip()
                if not line: # line is empty
                    if labeled_sentence: # labeled_sentence is not empty 
                        result.append(Parser.parse_foramtted_sentence(labeled_sentence))
                    labeled_sentence = []
                    continue
                
                labeled_sentence.append(line)
                
        if not result:
            print("Data is empty. Exiting program...")
            sys.exit()
        
        return result
    
    @staticmethod
    def parse_foramtted_sentence(labeled_sentence):
        root_token = LabeledToken(
                idx=0,
                token=ROOT_SYMBOL,
                pos=None,
                head=None
            )
        parsed_sentence = [root_token]
        
        for line in labeled_sentence:
            splitted_line = line.split('\t')
            labeled_token = LabeledToken(
                    idx=int(splitted_line[0]),
                    token=splitted_line[1],
                    pos=splitted_line[3],
                    head=int(splitted_line[6]) if splitted_line[6].isdigit() else None
                )
            parsed_sentence.append(labeled_token)
            
        return parsed_sentence
    
    def get_train_sentences(self):
        return self.train_sentences
        
