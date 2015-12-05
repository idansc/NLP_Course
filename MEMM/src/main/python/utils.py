import constants
import numpy as np

from math import exp

def calc_prob_denum(feat_manager, history, v):
    res = np.zeros(len(constants.TAGS))
    for i, tag in enumerate(constants.TAGS): 
        indices = feat_manager.calc_feature_vec(history, tag)
        if not indices:
            continue
        res[i] = sum(v[indices])
    
    return sum(exp(p) for p in res)

def calc_prob(feat_manager, v, history, tag):
    indices = feat_manager.calc_feature_vec(history, tag)
    denum = calc_prob_denum(feat_manager, history, v)
    if not indices:
        res = 1 / denum
    else:
        vec = np.zeros_like(v)
        vec[indices] = exp(sum(v[indices])) / denum
        res = sum(vec)
    
    return res