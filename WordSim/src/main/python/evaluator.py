from math import sqrt

class Evaluator(object):

    def __init__(self, parser):
        self.parser = parser
        self.matrices = [
                (parser.freqL1Mat, 'freq1'), 
                (parser.freqL2Mat, 'freq2'), 
                (parser.ppmiL1Mat, 'ppmi1'), 
                (parser.ppmiL2Mat, 'ppmi2')
            ]
        self.results = {'wordsim353': {}, 'simlex999': {}}
        
    def evaluate(self):
        print()
        print("Spearman Correlation:")
        for mat, name in self.matrices:
            self.calc_similarity_wordsim353(mat, name)
            self.calc_similarity_simlex999(mat, name)
            self.calc_correlation(name)
        print()
#         print(self.results)
    
    def calc_correlation(self, name):
        results_list = [(w1,w2,my_score,gt_score) for (w1,w2),(my_score,gt_score) in self.results['wordsim353'][name].items()]
        list_by_x = [(w1,w2) for w1,w2,_,_ in sorted(results_list, key=lambda e : e[2])]
        list_by_y = [(w1,w2) for w1,w2,_,_ in sorted(results_list, key=lambda e : e[3])]
        
        rank_table = []
        i = 1
        for p in list_by_x:
            rank_table.append((i, list_by_y.index(p) + 1))
            i += 1
        
        s = sum([pow(r_x-r_y,2) for (r_x, r_y) in rank_table])
        n = len(rank_table)
        rho = 1 - ((6*s)/(n*(pow(n,2)-1)))
        print(name+":", rho)
        
        
    def calc_similarity_wordsim353(self, mat, name):
        similarities = {}
        for (word1, word2), sim in self.parser.wordsim_db.items():
            if word1 not in mat.words or word2 not in mat.words or not mat.words[word1] or not mat.words[word2]:
                continue
            similarities[(word1, word2)] = (self.cosine(word1, word2, mat), sim.score)
        
        self.results['wordsim353'][name] = similarities
    
    def calc_similarity_simlex999(self, mat, name):
        similarities = {'A': {}, 'N': {}, 'V': {}}
        for (word1, word2), sim in self.parser.simlex_db.items():
            if word1 not in mat.words or word2 not in mat.words or not mat.words[word1] or not mat.words[word2]:
                continue
            similarities[sim.pos][(word1, word2)] = (self.cosine(word1, word2, mat), sim.score)

        self.results['simlex999'][name] = similarities
    
    def cosine(self, word1, word2, mat):
        v1 = mat.words[word1]
        v2 = mat.words[word2]
        norm_factor = sqrt(sum([pow(f,2) for f in v1.values()])) * sqrt(sum([pow(f,2) for f in v2.values()]))
        words = v1.keys() | v2.keys()
        num = 0.0
        for word in words:
            num += v1.get(word, 0.0) * v2.get(word, 0.0)
        
        return num / norm_factor
        
        
        