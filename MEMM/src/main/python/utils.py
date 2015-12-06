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
    denum = calc_prob_denum(feat_manager, history, v)
    
    indices = feat_manager.calc_feature_vec(history, tag)
    if not indices:
        res = 1.0 / denum
    else:
        res = exp(sum(v[indices])) / denum
    
    return res