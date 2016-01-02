import os
import numpy as np

from edmonds import edmonds

class Inferrer(object):
    '''
    Infers a parsing tree of a sentence by finding an MST.
    '''

    def __init__(self, in_file_path, parser, features_manager, w):
        self.parsed_data = parser.parse_foramtted_data(in_file_path)
        self.dep_trees = []
        self.infile_ext = os.path.splitext(in_file_path)[-1]

        for sentence in self.parsed_data:
            G = self.build_graph(sentence, features_manager, w)
            root = 0
            self.dep_trees.append(edmonds.mst(root, G))
#         print(self.dep_trees)
    
    def print_statistics(self):
        if self.infile_ext != ".labeled":
            raise Exception("Cannot extract statistics from an unlabeled file.")
        
        hits = 0
        tokens = 0
        
        for i, sentence in enumerate(self.parsed_data):
            dep_tree = Inferrer.reformat_dep_tree(self.dep_trees[i])
            for labeled_token in sentence[1:]:
                if dep_tree[labeled_token.idx] == labeled_token.head:
                    hits += 1
                tokens += 1
            
#             print("Sentence No.", i, " ; Current Accuracy:", hits / tokens)
        
        print("\n   **** Statistics ****   \n")
        print("Number of sentences:", len(self.parsed_data))
        print("Number of tokens:", tokens)
        print("Number of hits:", hits)
        print("Accuracy:", hits / tokens)

    
    def store(self, out_file_path):
        with open(out_file_path, 'w') as outfile:
            for i, sentence in enumerate(self.parsed_data):
                sentence_str = self.create_sentence_str(sentence, i)
                print(sentence_str, end="\n", file=outfile)
    
    def create_sentence_str(self, sentence, i):
        dep_tree = Inferrer.reformat_dep_tree(self.dep_trees[i])
        outstr = ""
        for labeled_token in sentence[1:]:
            outstr += str(labeled_token.idx)+"\t"+labeled_token.token+"\t_\t"+labeled_token.pos+"\t_\t_\t"+str(dep_tree[labeled_token.idx])+"\t_\t_\t_\n"
        
        return outstr    
    
    @staticmethod
    def reformat_dep_tree(dep_tree):
        res = {}
        for key, value in dep_tree.items():
            for inner_key in value.keys():
                res[inner_key] = key
        
        return res
            
    @staticmethod
    def build_graph(sentence, features_manager, w):
        n = len(sentence)
        G = {}
        for i in range(n):
            G[i] = {j:-np.dot(features_manager.calc_feature_vec(sentence,i,j),w) for j in range(1, n) if j != i}
        return G   
        