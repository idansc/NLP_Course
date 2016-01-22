from nltk.stem.wordnet import WordNetLemmatizer
from constants import BOUNDARY_SYMBOL
from collections import Counter

class TermContextMatrix(object):

    def __init__(self):
        self.contexts = Counter()
        self.words = {}
        self.lmtzr = WordNetLemmatizer()
        self.N = 0
        self.non_zeros_entries = 0
    
    def add_word_with_context(self, word, contexts):
        word = self.lmtzr.lemmatize(word.lower())
        contexts = [self.lmtzr.lemmatize(w.lower()) for w in contexts if w != BOUNDARY_SYMBOL]
        
        row = self.words.get(word, Counter())      
        for ctx in contexts:
            self.contexts[ctx] += 1
            row[ctx] += 1
        
        self.words[word] = row
        
    def filter(self):
        self.contexts = dict(self.contexts.most_common(5000))
        for _, row in self.words.items(): 
            to_delete = []
            for ctx in row:
                if ctx not in self.contexts:
                    to_delete.append(ctx)
            for ctx in to_delete:
                del row[ctx]
            self.N += sum(row.values())
            self.non_zeros_entries += len(row.values())

    def clac_sparsity(self):
        return self.non_zeros_entries / (len(self.contexts)*len(self.words))