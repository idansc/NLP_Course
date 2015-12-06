import numpy as np
import utils
from constants import IGNORE_WORDS, START_SYMBOL
from itertools import product
from history import History
from math import log


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
        pi[(1, START_SYMBOL, START_SYMBOL)] = log(1 + 1)
        history = History()
        #algo
        for k in range(2, len(sentence)-1):
            t_tags = self.s_k(sentence[k-2])
            u_tags = self.s_k(sentence[k-1])
            v_tags = self.s_k(sentence[k])
            for u_idx, v_idx in product(range(len(u_tags)), range(len(v_tags))):
                pi_vals = []
                for t_idx in range(len(t_tags)):
                    tm2 = t_tags[t_idx]
                    tm1 = u_tags[u_idx]
                    w = sentence[k]
                    wm1 = sentence[k-1]
                    wp1 = sentence[k+1] # That's why need $ (i.e. STOP sign) at the end of each sentence
                    history.set(tm2, tm1, wm1, w, wp1)
                    q_val = utils.calc_prob(self.feat_manager,self.v, history,v_tags[v_idx])
                    key = (k-1,t_tags[t_idx],u_tags[u_idx])
                    prev_pi = pi[key]

                    pi_val= prev_pi + log(q_val+1)
                    pi_vals.append(pi_val)
                max_index = np.argmax(pi_vals)
                bp[(k,u_tags[u_idx],v_tags[v_idx])] = t_tags[max_index]
                pi[(k,u_tags[u_idx],v_tags[v_idx])] = pi_vals[max_index]
        t = []
        u_tags = self.s_k(sentence[-3])
        v_tags = self.s_k(sentence[-2])
        uv_tags = list(product(range(len(u_tags)), range(len(v_tags))))
        del pi_vals[:]
        for u,v in uv_tags:
            key = (len(sentence)-2,u_tags[u],v_tags[v])
            pi_vals.append(pi[key])
        u_max, v_max = uv_tags[np.argmax(pi_vals)]
        t.insert(0,v_tags[v_max])
        t.insert(0,u_tags[u_max])
        for k in range(len(sentence)-4,1,-1):
            t.insert(0,bp[(k+2,t[0],t[1])])
        return t




