import sys

from constants import START_SYMBOL, END_SYMBOL, TAGS, IGNORE_WORDS, IGNORE_TAGS, ACTUAL_TAGS
from collections import defaultdict
from collections import Counter

class Parser(object):
    '''
    Handles parsing and related aspects
    '''
    
    START_TUPLE = (START_SYMBOL, START_SYMBOL)
    END_TUPLE = (END_SYMBOL, END_SYMBOL)
    
    def __init__(self, training_data_path, test_data_path, viterbi_tags_treshold, use_common_tags):
        self.sentences = []
        self.test_sentences = []
        self.parse_training(training_data_path)
        self.viterbi_tags_treshold = viterbi_tags_treshold
        self.use_common_tags = use_common_tags
        self.parse_test(test_data_path)
        self.count_tags = Counter()
        self.word_tags_dict = defaultdict(set)
        self.init_word_tags()

    def parse_training(self,filepath):
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

    def parse_test(self, test_data_path):
        with open(test_data_path, 'r') as test_file:
            for line in test_file:
                line = line.strip()
                sentence = \
                    [START_SYMBOL, START_SYMBOL] \
                    + [tuple(w.split('_'))[0] for w in line.split(' ')] \
                    + [END_SYMBOL]
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
            for _, t in s[2:-1]:
                if t in IGNORE_TAGS:
                    continue
                self.count_tags[t]+=1

    def init_word_tags(self):
        self.init_common_tags()
        for s in self.sentences:
            for (w, t) in s[2:-1]:
                if w in IGNORE_WORDS:
                    continue
                self.word_tags_dict[w].add(t)
            # fit the tags to threshold
            if self.use_common_tags == True:
                all_words = self.get_all_words()
                most_common_tags_pairs = self.count_tags.most_common(self.viterbi_tags_treshold)
                most_common_tags = [pair[0] for pair in most_common_tags_pairs]
                for w in all_words:
                    self.word_tags_dict[w].update(most_common_tags)
                    '''   thresh_diff = self.viterbi_tags_treshold - len(self.word_tags_dict[w])
                        while thresh_diff > 0:
                            for tag in self.count_tags.most_common(self.viterbi_tags_treshold):
                                if tag not in self.word_tags_dict[w]:
                                    thresh_diff-=1
                                    self.word_tags_dict[w].add(tag)
                    '''


    def get_word_tags(self, w):
        #returns the tags for a word, the default for unknown word is top tags.
        #return self.word_tags_dict.get(w,self.count_tags.most_common(self.viterbi_tags_treshold))
        return self.word_tags_dict.get(w, ACTUAL_TAGS)
        