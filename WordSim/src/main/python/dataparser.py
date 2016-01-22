import re
from math import log
from similarity import Similarity
from matrix import TermContextMatrix
from constants import *
from utils import *

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
        self.ppmiL1Mat = Parser.to_ppmi(self.freqL1Mat)
        self.ppmiL2Mat = Parser.to_ppmi(self.freqL2Mat)
        
        print()
        print('Sparsity')
        print('freq1:', self.freqL1Mat.clac_sparsity())
        print('freq2:', self.freqL2Mat.clac_sparsity())
        print('ppmi1:', self.ppmiL1Mat.clac_sparsity())
        print('ppmi2:', self.ppmiL2Mat.clac_sparsity())
        print()
#         print("N:", self.freqL1Mat.N, self.ppmiL1Mat.N, self.freqL2Mat.N, self.ppmiL2Mat.N)
#         print("len(contexts):", len(self.freqL1Mat.contexts), len(self.ppmiL1Mat.contexts), len(self.freqL2Mat.contexts), len(self.ppmiL2Mat.contexts))
#         print("len(words):", len(self.freqL1Mat.words), len(self.ppmiL1Mat.words), len(self.freqL2Mat.words), len(self.ppmiL2Mat.words))
#         print(self.freqL1Mat.words)
#         print(self.ppmiL1Mat.words)
    
    @staticmethod
    def to_ppmi(freq_mat):
        ppmi_mat = TermContextMatrix()
        ppmi_mat.contexts = freq_mat.contexts
        ppmi_mat.lmtzr = freq_mat.lmtzr
        
        contexts_size = len(freq_mat.contexts)
        words_size = len(freq_mat.words)
        ppmi_mat.N = freq_mat.N + (contexts_size*words_size*2)
        ppmi_mat.words = {}
        ppmi_mat.non_zeros_entries = 0
        
        for word, row in freq_mat.words.items(): 
            p_i = (sum(row.values()) + 2*contexts_size) / ppmi_mat.N
            
            ctx_prob_pairs = []
            for ctx_word, cnt in row.items():
                p_j = (freq_mat.contexts[ctx_word] + 2*words_size) / ppmi_mat.N
                p_ij = (cnt + 2) / ppmi_mat.N
                pmi = log(p_ij / (p_i * p_j), 2)
                ppmi = pmi if pmi > 0 else 0.0
                ctx_prob_pairs.append((ctx_word, ppmi))
                if ppmi != 0.0:
                    ppmi_mat.non_zeros_entries += 1
                
            ppmi_mat.words[word] = dict(ctx_prob_pairs)
        
        return ppmi_mat
            
    def calculate_freq_matrices(self, corpus):
        prep_time = time.time()
        prep_partial_time = prep_time
        with open(corpus, errors='ignore') as file:
            for j,line in enumerate(file):
                if j % 10000 == 0:
                    print("Current line:", j, "; Elapsed time:", calc_elpased_time(prep_partial_time))
                    prep_partial_time = time.time()
                line = re.sub(r'([^\sA-Za-z0-9]|_)', '', line.strip())
                line = re.sub(r'\b\d\d\d\d\b', YEAR_SYMBOL, line.strip())
                line = re.sub(r'\d+', NUMBER_SYMBOL, line)
                splitted_line = [BOUNDARY_SYMBOL, BOUNDARY_SYMBOL] + line.split() + [BOUNDARY_SYMBOL, BOUNDARY_SYMBOL]
                for i, word in enumerate(splitted_line[2:-2]):
                    if word in self.words:
                        i += 2
                        self.freqL1Mat.add_word_with_context(word, [splitted_line[i-1], splitted_line[i+1]])
                        self.freqL2Mat.add_word_with_context(word, [splitted_line[i-2], splitted_line[i-1], splitted_line[i+1], splitted_line[i+2]])
        
        print("Done preprocessing. Elapsed time:", calc_elpased_time(prep_time))
        
        filter_time = time.time()
        self.freqL1Mat.filter()
        self.freqL2Mat.filter()
        print("Done filtering. Elapsed time:", calc_elpased_time(filter_time))
        
    
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
                