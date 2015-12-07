
from inference import Inference
from collections import Counter

class Statistics(object):
    '''
    Extracts statistics and results for the MEMM tagger
    '''

    def __init__(self, parser, v, feat_manager, config):
        self.test_sentences = parser.get_test_sentences()
        self.inference = Inference(parser, v, feat_manager)
        self.parser = parser
        self.confusion_matrix_print = config["confusion_matrix_print"]
        self.compatition = config['competition']

        self.test_sentences_words_only = []
        for s in parser.get_test_sentences():
            self.test_sentences_words_only.append([wt[0] for wt in s])
        
    def print_statistics(self):

        total_unknown_words = 0
        hits_unknown_words = 0

        total_words = 0
        hits = 0

        missed_tags = Counter()
        predicted = Counter()
        if self.compatition:
            for i, s in enumerate(self.parser.get_comp_sentences()):
                tags = self.inference.viterbi(s)
                actual_s = s[2:-1]
                tagged_sentence = []
                for j in range(len(actual_s)):
                    tagged_sentence.append(actual_s[j]+"_"+tags[j])
                with open("result.txt", "w") as f:
	                f.write(" ".join(tagged_sentence))

        else:
            for i, s in enumerate(self.test_sentences_words_only):
                tags = self.inference.viterbi(s)
                actual_s = s[2:-1]
                tagged_sentence = []

                if len(actual_s) != len(tags):
                    raise Exception("Incompatible lengths")

                s_tagged = list(zip(actual_s, tags))
                total_words += len(s_tagged)
                ground_truth = self.test_sentences[i][2:-1]

                for j, wt in enumerate(s_tagged):
                    if wt == ground_truth[j]:
                        hits += 1
                    else:
                        missed_tags[ground_truth[j][1]] += 1
                    predicted[(ground_truth[j][1], wt[1])]+=1
                    if not self.parser.is_word_known(wt[0]):
                        if wt == ground_truth[j]:
                            hits_unknown_words += 1
                        total_unknown_words +=1
                print("Sentence No.", i, " ; Current Accuracy:", hits / total_words, " ; Current Unknown Words Accuracy:", hits_unknown_words / total_unknown_words)

            print("\n   **** Statistics ****   \n")
            print("Total amount of sentences:", len(self.test_sentences_words_only))
            print("Total amount of words:", total_words)
            print("Total amount of hits:", hits)
            print("Accuracy:", hits / total_words)
            print("Total amount of unknown words:", total_unknown_words)
            print("Total amount of unknown words hits:", hits_unknown_words)
            print("Unknown Words accuracy:", hits_unknown_words / total_unknown_words)
            print("\n ****Most missed tags**** \n")
            if(self.confusion_matrix_print):
                most_missed_tags = [pairs[0] for pairs in missed_tags.most_common(10)]
                print(missed_tags.most_common(10))
                for tag1 in most_missed_tags:
                    for tag2 in most_missed_tags:
                        print("Actual: ", tag1, "Predict:", tag2, "amount of times:", predicted.get((tag1,tag2),0))