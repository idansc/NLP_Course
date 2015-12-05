import numpy as np
import utils
from constants import IGNORE_WORDS, START_SYMBOL
from itertools import product
from history import History

class Inference:
    def __init__(self, parser, v, feat_manager):
        self.parser = parser
        self.v = v
        self.feat_manager = feat_manager

    def s_k(self,w):
        if w in IGNORE_WORDS:
            return [w]
        return list(self.parser.get_word_tags(w))

    def viterbi(self, sentence):
        pi = {}
        bp = {}
        #init
        pi[(1, START_SYMBOL, START_SYMBOL)] = 1
        history = History()
        #algo
        for k in range(2, len(sentence)):
            t_tags = self.s_k(sentence[k-2])
            u_tags = self.s_k(sentence[k-1])
            v_tags = self.s_k(sentence[k])
            for u_idx, v_ind in product(range(len(u_tags)), range(len(v_tags))):
                pi_vals = []
                for t_idx in range(len(t_tags)):
                    # TODO: we should add qval
                    
                    # -----------------------------
                    
#                     tm2 = ...
#                     tm1 = ...
#                     w = sentence[k]
#                     wm1 = sentence[k-1]
#                     wp1 = sentence[k+1] # That's why need $ (i.e. STOP sign) at the end of each sentence
#                     history.set(tm2, tm1, wm1, w, wp1)
#                     
#                     qval = utils.calc_prob(self.feat_manager, history, self.v, index=k)
                    
                    # -----------------------------
                    
                    qval = 1
                    key = (k-1,t_tags[t_idx],u_tags[u_idx])
                    prev_pi = pi[key]
                    #TODO: maybe should be with log and similiar.
                    pi_val= prev_pi*qval
                    pi_vals.append(pi_val)
                max_index = int(np.argmax(pi_vals))
                bp[(k,u_tags[u_idx],v_tags[v_ind])] = t_tags[max_index]
                pi[(k,u_tags[u_idx],v_tags[v_ind])] = pi_vals[max_index]
        t = []
        u_tags = self.s_k(sentence[-2])
        v_tags = self.s_k(sentence[-1])
        uv_tags = list(product(range(len(u_tags)), range(len(v_tags))))
        del pi_vals[:]
        for u,v in uv_tags:
            key = (len(sentence)-1,u_tags[u],v_tags[v])
            pi_vals.append(pi[key])
        u_max, v_max = uv_tags[np.argmax(pi_vals)]
        t.insert(len(sentence)-1,u_tags[u_max])
        t.insert(len(sentence)-2,v_tags[v_max])
        for k in range(len(sentence)-3,1,-1):
            t.insert(0,bp[(k+2,t[0],t[1])])
        return t




