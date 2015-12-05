import sys

from utils import START_SYMBOL, END_SYMBOL, TAGS, IGNORE_WORDS, IGNORE_TAGS
from collections import defaultdict
from collections import Counter

class Parser(object):
    '''
    Handles parsing and related aspects
    '''
    
    START_TUPLE = (START_SYMBOL, START_SYMBOL)
    END_TUPLE = (END_SYMBOL, END_SYMBOL)
    VITERBI_TAGS_THRESHOLD = 6
    
    def __init__(self, filepath, test_fp):
        self.sentences = []
        self.test_sentences = []
        self.parse(filepath)
        self.parse_test(test_fp)
        self.count_tags = Counter()
        self.word_tags_dict = defaultdict(set)
        self.init_word_tags()

    def parse(self,filepath):
        with open(filepath, 'r') as datafile:
            for line in datafile:
                line = line.strip()
                sentence = \
                    [self.START_TUPLE, self.START_TUPLE] \
                    + [tuple(w.split('_')) for w in line.split(' ')] \
                    + [self.END_TUPLE]
                
                self.sentences.append(sentence)
        
        if len(self.sentences) == 0:
            print("Training data is empty. Exiting program...")
            sys.exit()

    def parse_test(self,test_fp):
        with open(test_fp, 'r') as train_file:
            for line in train_file:
                line = line.strip()
                sentence = \
                    [START_SYMBOL, START_SYMBOL] \
                    + [tuple(w.split('_'))[0] for w in line.split(' ')]
            self.test_sentences.append(sentence)

    def get_test_sentences(self):
        return self.test_sentences

    def get_sentences(self):
        return self.sentences
    
    def get_all_tags(self):
        tags = set()
        for s in self.sentences:
            for wt in s:
                tags.add(wt[1])
        
        return tags

    def get_all_words(self):
        tags = set()
        for s in self.sentences:
            for wt in s:
                tags.add(wt[0])
        
        return tags
    
    def get_num_words(self):
        return sum(len(s) - 3 for s in self.sentences)

    def init_common_tags(self):
        for s in self.sentences:
            for i, (w, t) in enumerate(s[2:-1]):
                if t in IGNORE_TAGS:
                    continue
                self.count_tags[t]+=1

    def init_word_tags(self):
        self.init_common_tags()
        for s in self.sentences:
            for i, (w, t) in enumerate(s[2:-1]):
                if w in IGNORE_WORDS:
                    continue
                self.word_tags_dict[w].add(t)
            # fit the tags to threshold
            all_words = self.get_all_words()
            must_common_tags_pairs = self.count_tags.most_common(self.VITERBI_TAGS_THRESHOLD)
            must_common_tags = [pair[0] for pair in must_common_tags_pairs]
            for w in all_words:
             self.word_tags_dict[w].update(must_common_tags)
             '''   thresh_diff = self.VITERBI_TAGS_THRESHOLD - len(self.word_tags_dict[w])
                while thresh_diff > 0:
                    for tag in self.count_tags.most_common(self.VITERBI_TAGS_THRESHOLD):
                        if tag not in self.word_tags_dict[w]:
                            thresh_diff-=1
                            self.word_tags_dict[w].add(tag)
             '''


    def get_word_tags(self,w):
        #returns the tags for a word, the default for unknown word is top tags.
       #return self.word_tags_dict.get(w,self.count_tags.most_common(self.VITERBI_TAGS_THRESHOLD))
        return self.word_tags_dict.get(w,TAGS)
        