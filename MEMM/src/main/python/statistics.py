
from inference import Inference
from collections import Counter

class Statistics(object):
    '''
    Extracts statistics and results for the MEMM tagger
    '''

    def __init__(self, parser, v, feat_manager):
        self.test_sentences = parser.get_test_sentences()
        self.inference = Inference(parser, v, feat_manager)
        self.parser = parser

        self.test_sentences_words_only = []
        for s in parser.get_test_sentences():
            self.test_sentences_words_only.append([wt[0] for wt in s])
        
    def print_statistics(self):

        total_unknown_words = 0
        hits_unknown_words = 0

        total_words = 0
        hits = 0

        missed_tags = Counter()
        
        for i, s in enumerate(self.test_sentences_words_only):
            tags = self.inference.viterbi(s)
            actual_s = s[2:-1]
            
            if len(actual_s) != len(tags):
                raise Exception("Incompatible lengths")
            
            s_tagged = list(zip(actual_s, tags))
            total_words += len(s_tagged)
            ground_truth = self.test_sentences[i][2:-1]
            
            for j, wt in enumerate(s_tagged):
                if wt == ground_truth[j]:
                    hits += 1
                else:
                    missed_tags[wt] += 1
                if not self.parser.is_word_known(actual_s[j]):
                    if wt == ground_truth[j]:
                        hits_unknown_words += 1
                    total_unknown_words +=1
            print("Sentence No.", i, " ; Current Accuracy:", hits / total_words, " ; Current Unknown Words Accuracy:", hits_unknown_words / total_unknown_words)
        
        print("\n   **** Statistics ****   \n")
        print("Total amount of sentences:", len(self.test_sentences_words_only))
        print("Total amount of words:", total_words)
        print("Total amount of hits:", hits)
        print("Accuracy:", hits / total_words)
        print("Total amount of unknown words:", total_words)
        print("Total amount of unknown words hits:", hits_unknown_words)
        print("Unknown Words accuracy:", hits_unknown_words / total_unknown_words, "\n")

