import sys

class Parser(object):
    '''
    Handles data parsing
    '''

    def __init__(self, training_data_path):
        self.sentences = []
        self.test_sentences = self.parse_labeled_data(training_data_path)
        
        
    def parse_labeled_data(self, filepath):
        result = []
        with open(filepath, 'r') as datafile:
            labeled_sentence = []
            for line in datafile:
                line = line.strip()
                if not line: # list is empty
                    if labeled_sentence: # labeled_sentence is not empty 
                        result.append(self.parse_labeled_sentence(labeled_sentence))
                    labeled_sentence = []
                    continue
                
                labeled_sentence.append(line)
                
        if not result:
            print("Labeled data is empty. Exiting program...")
            sys.exit()
        
        return result
    
    def parse_labeled_sentence(self, labeled_sentence):
        parsed_sentence = []
        for labeled_word in labeled_sentence:
            labeled_word_array = labeled_word.split('\t')
            parsed_sentence.append((labeled_word_array[0], labeled_word_array[1], labeled_word_array[3], labeled_word_array[6]))
        return parsed_sentence
        
