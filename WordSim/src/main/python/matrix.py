from collections import OrderedDict

class TermContextMatrix(object):

    def __init__(self):
        self.contexts = []
        self.words = {}
    
    def add_word_with_context(self, word, contexts):
        row = self.words.get(word, {})
        for context in contexts:
            if context not in self.contexts:
                self.contexts.append(context)
            idx = self.contexts.index(context)
            row[idx] = row.get(idx, 0) + 1
        
        self.words[word] = row