import sys

from utils import START_SYMBOL, END_SYMBOL
from collections import defaultdict
from collections import Counter

class Parser(object):
    '''
    Handles parsing and related aspects
    '''
    
    START_TUPLE = (START_SYMBOL, START_SYMBOL)
    END_TUPLE = (END_SYMBOL, END_SYMBOL)
    VITERBI_TAGS_THRESHOLD = 6
    
    def __init__(self, filepath):
        self.sentences = []
        self.parse(filepath)
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
                
                self.word_tag_array += parsed_phrased
            
            self.word_tag_array += [self.EOF_TUPLE]
        
        if len(self.word_tag_array) == 0:
            print("Training data is empty. Exiting program...")
            sys.exit()
    
    def get_word_tag_array(self):
        return self.word_tag_array
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
                tags.add(wt[1])
        
        return tags
    
    def get_num_words(self):
        return sum(len(s) - 3 for s in self.sentences)

    def init_common_tags(self):
        for t in self.get_all_tags():
            self.count_tags[t]+=1

    def init_word_tags(self):
        self.init_common_tags()
        for w,t in self.word_tag_array:
            if t == START_SYMBOL or t == END_SYMBOL or t==EOF_SYMBOL or t =='``' or t=="." or t=="," or t==":" or t=="(" or t==")" or t=="''" or t =='``':
                continue
            self.word_tags_dict[w].add(t)
            #maybe it will be better to threshold here as well.

            '''
                fit the tags to threshold
            '''
            '''
            #this part should remove seen tags - in case vitarbi will be too slow.
            while thresh_diff < 0:
                for tag in self.count_tags.(self.VITERBI_TAGS_THRESHOLD):
                    if tag not in self.word_tags_dict[w]:
                       thresh_diff-=1
                       self.word_tags_dict[w].append(tag)
            '''
        for w in self.get_all_words():
            thresh_diff = self.VITERBI_TAGS_THRESHOLD - len(self.word_tags_dict[w])
            while thresh_diff > 0:
                for tag in self.count_tags.most_common(self.VITERBI_TAGS_THRESHOLD):
                    if tag not in self.word_tags_dict[w]:
                        thresh_diff-=1
                        self.word_tags_dict[w].add(tag)



    def get_word_tags(self,w):
        #returns the tags for a word, the default for unknown word is top tags.
       #return self.word_tags_dict.get(w,self.count_tags.most_common(self.VITERBI_TAGS_THRESHOLD))
        return self.word_tags_dict.get(w,set(TAGS))
        