import sys

from labeled_token import LabeledToken
from constants import ROOT_SYMBOL
from collections import Counter


class Parser(object):
    '''
    Handles data parsing
    '''

    def __init__(self, training_data_path, prefix_flag, prefix_threshold):
        self.train_sentences = self.parse_foramtted_data(training_data_path)
        if prefix_flag:
            self.prefixes = Counter()
            self.prefix_threshold = prefix_threshold
            self.set_prefixes_data(prefix_threshold)




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
        
        for labeled_token in parsed_sentence[1:]:
            labeled_token.in_between = Parser.get_in_between_tags(parsed_sentence,
                                                                  labeled_token.head,
                                                                  labeled_token.idx)
            
        return parsed_sentence
    
    @staticmethod
    def get_in_between_tags(sentence, head, modifier):
        min_idx = min(head, modifier)
        max_idx = max(head, modifier)
        result = [];
        for token in sentence[min_idx+1:max_idx]:
            result.append(token.pos)
        return result

    def set_prefixes_data(self,prefix_threshold):
        for sentence in self.get_train_sentences():
            for labeled_token in sentence[1:]:
                if len(labeled_token.token) > prefix_threshold:
                    self.prefixes[labeled_token.token[:prefix_threshold]]+=1
    
    def get_train_sentences(self):
        return self.train_sentences

    def get_prefix(self):
        return {pair[0] for pair in self.prefixes.items() if pair[1] >= self.prefix_threshold}
        
