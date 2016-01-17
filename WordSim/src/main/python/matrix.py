from nltk.stem.wordnet import WordNetLemmatizer
from constants import BOUNDARY_SYMBOL

class TermContextMatrix(object):

    def __init__(self):
        self.contexts = []
        self.words = {}
        self.lmtzr = WordNetLemmatizer()
    
    def add_word_with_context(self, word, contexts):
        word = self.lmtzr.lemmatize(word.lower())
        contexts = [self.lmtzr.lemmatize(w.lower()) for w in contexts if w != BOUNDARY_SYMBOL]
        
        row = self.words.get(word, {})
        for context in contexts:
            if context not in self.contexts:
                self.contexts.append(context)
            idx = self.contexts.index(context)
            row[idx] = row.get(idx, 0) + 1
        
        self.words[word] = row
    
