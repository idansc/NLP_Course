from similarity import Similarity
from matrix import TermContextMatrix

class Parser(object):
    '''
    Handles data parsing
    '''

    def __init__(self, simlex_path, wordsim_path, corpus):
        self.words = set()
        self.freqL1Mat = TermContextMatrix()
        self.freqL2Mat = TermContextMatrix()
        self.simlex_db = self.parse_sim_db(simlex_path)
        self.wordsim_db = self.parse_sim_db(wordsim_path)
        self.calculate_freq_matrices(corpus)
        
        print(self.freqL2Mat.contexts)
        print(self.freqL2Mat.words)
    
    def calculate_freq_matrices(self, curpos):
        BOUNDARY_SYMBOL = '*'
        with open(curpos, 'r') as file:
            for line in file:
                splitted_line = [BOUNDARY_SYMBOL, BOUNDARY_SYMBOL] + line.strip().split() + [BOUNDARY_SYMBOL, BOUNDARY_SYMBOL]
                for i, word in enumerate(splitted_line[2:-2]):
                    if word in self.words:
                        i += 2
                        self.freqL1Mat.add_word_with_context(word, [splitted_line[i-1], splitted_line[i+1]])
                        self.freqL2Mat.add_word_with_context(word, [splitted_line[i-2], splitted_line[i-1], splitted_line[i+1], splitted_line[i+2]])
    
    def parse_sim_db(self, db_path):
        result = {}
        with open(db_path, 'r') as file:
            next(file)
            for line in file:
                splitted_line = line.strip().split(',')
                k = frozenset({splitted_line[0], splitted_line[1]})
                v = Similarity(float(splitted_line[2]), splitted_line[3]) if len(splitted_line) == 4 else Similarity(float(splitted_line[2]))
                result[k] = v
                self.words.add(splitted_line[0])
                self.words.add(splitted_line[1])
            
        return result       
                