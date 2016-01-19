from nltk.stem.wordnet import WordNetLemmatizer
from constants import BOUNDARY_SYMBOL
from collections import Counter

class TermContextMatrix(object):

    def __init__(self):
        self.contexts = Counter()
        self.words = {}
        self.lmtzr = WordNetLemmatizer()
    
    def add_word_with_context(self, word, contexts):
        word = self.lmtzr.lemmatize(word.lower())
        contexts = [self.lmtzr.lemmatize(w.lower()) for w in contexts if w != BOUNDARY_SYMBOL]
        
        row = self.words.get(word, Counter())      
        for ctx in contexts:
            self.contexts[ctx] += 1
            row[ctx] += 1
        
        self.words[word] = row
        
    def filter(self):
        self.contexts = [ctx for ctx,_ in self.contexts.most_common(5000)]