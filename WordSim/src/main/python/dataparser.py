from similarity import Similarity

class Parser(object):
    '''
    Handles data parsing
    '''

    def __init__(self, simlex_path, wordsim_path, corpus):
        self.words = set()
        self.simlex_db = self.parse_sim_db(simlex_path)
        self.wordsim_db = self.parse_sim_db(wordsim_path)
    
    def parse_sim_db(self, db_path):
        result = {}
        with open(db_path, 'r') as file: #TODO skip first line
            next(file)
            i = 0
            for line in file:
                line = line.strip()
                splitted_line = line.split(',')
                k = frozenset({splitted_line[0], splitted_line[1]})
                v = Similarity(float(splitted_line[2]), splitted_line[3]) if len(splitted_line) == 4 else Similarity(float(splitted_line[2]))
                result[k] = v
                self.words.add(splitted_line[0])
                self.words.add(splitted_line[1])
                i += 1
            
        return result       
                