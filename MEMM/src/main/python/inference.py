import numpy as np
from utils import IGNORE_WORDS, START_SYMBOL
from math import log
from itertools import product
from dataparser import Parser


class Inference:
    def __init__(self,parser):
        self.parser = parser

    def s_k(self,w):
        if w in IGNORE_WORDS:
            return [w]
        return list(self.parser.get_word_tags(w))

    def viterbi(self,s):
        pi = {}
        bp = {}
        #init
        pi[(1, START_SYMBOL, START_SYMBOL)] = 1
        #algo
        for k in range(2, len(s)):
            t_tags = self.s_k(s[k-2])
            u_tags = self.s_k(s[k-1])
            v_tags = self.s_k(s[k])
            for u_ind, v_ind in product(range(len(u_tags)), range(len(v_tags))):
                pi_vals = []
                for t_ind in range(len(t_tags)):
                    # TODO: we should add qval
                    qval = 1
                    key = (k-1,t_tags[t_ind],u_tags[u_ind])
                    prev_pi = pi[key]
                   #TODO: maybe should be with log and similiar.
                    pi_val= prev_pi*qval
                    pi_vals.append(pi_val)
                max_index = int(np.argmax(pi_vals))
                bp[(k,u_tags[u_ind],v_tags[v_ind])] = t_tags[max_index]
                pi[(k,u_tags[u_ind],v_tags[v_ind])] = pi_vals[max_index]
        t = []
        u_tags = self.s_k(s[-2])
        v_tags = self.s_k(s[-1])
        uv_tags = list(product(range(len(u_tags)), range(len(v_tags))))
        del pi_vals[:]
        for u,v in uv_tags:
            key = (len(s)-1,u_tags[u],v_tags[v])
            pi_vals.append(pi[key])
        u_max, v_max = uv_tags[np.argmax(pi_vals)]
        t.insert(len(s)-1,u_tags[u_max])
        t.insert(len(s)-2,v_tags[v_max])
        for k in range(len(s)-3,1,-1):
            t.insert(0,bp[(k+2,t[0],t[1])])
        return t




