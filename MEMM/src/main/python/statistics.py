
from inference import Inference

class Statistics(object):
    '''
    Extracts statistics and results for the MEMM tagger
    '''

    def __init__(self, parser, v, feat_manager):
        self.test_sentences = parser.get_test_sentences()
        self.inference = Inference(parser, v, feat_manager)
        
        self.test_sentences_words_only = []
        for s in parser.get_test_sentences():
            self.test_sentences_words_only.append([wt[0] for wt in s])
        
    def print_statistics(self):
        
        total_words = 0
        hits = 0
        
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
            print("Sentence No.", i, " ; Current accuracy:", hits / total_words)
            
        print("   **** Statistics ****   ")
        print("Total amount of sentences:", len(self.test_sentences_words_only))
        print("Total amount of words:", total_words)
        print("Total amount of hits:", hits)
        print("Accuracy:", hits / total_words)
