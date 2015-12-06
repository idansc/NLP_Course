import constants
import utils
import numpy as np
import time

from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from math import log, exp

from history import History

optimizer = None
    
class Optimizer(object):
    '''
    Optimizes the parameters of MEMM for a given set of features.
    '''

    def __init__(self, sentences, num_words, feat_manager, lambda_param, maxiter):
        self.lambda_param = lambda_param
        self.n = num_words
        self.m = feat_manager.get_num_features()
        self.sentences = sentences
        self.feat_manager = feat_manager
        self.maxiter = maxiter

        self.feat_metrix = self.clac_features_matrix()
        self.term1_of_loss_function_der = sum(self.feat_metrix)
        
        global optimizer
        optimizer = self
    
    def clac_features_matrix(self):
        history = History()
        col = np.array([], dtype=np.int)
        row = np.array([], dtype=np.int)

        idx = 0
        for s in self.sentences:
            for i, (word, tag) in enumerate(s[2:-1]):
#                 if word in [constants.START_SYMBOL, constants.END_SYMBOL, constants.DOT]:
#                     continue
                
                i += 2
                tm2 = s[i-2][1]
                tm1 = s[i-1][1]
                wm1 = s[i-1][0]
                wp1 = s[i+1][0]
                history.set(tm2, tm1, wm1, word, wp1)
                
                feat_vec = np.array(self.feat_manager.calc_feature_vec(history, tag))
                col = np.concatenate((col, feat_vec))
                row = np.concatenate((row, np.full(feat_vec.size, idx + i-2, dtype=np.int)))
            
            idx += len(s) - 3
                
        data = np.ones(col.size, dtype=np.int)
                
        return csr_matrix((data, (row, col)), shape=(self.n, self.m), dtype=np.int)
    
    def optimize(self, v0):
#         res = minimize(loss_function, v0, method='BFGS', jac=loss_function_der, options={'disp': True})
#         res = minimize(loss_function, v0, method='L-BFGS-B', jac=loss_function_der, options={'disp': True})
        res = minimize(loss_function, v0, method='L-BFGS-B', jac=loss_function_der, options={'maxiter': self.maxiter, 'disp': True})
        return res.x
        
    def calc_expected_counts_aux(self, sentence ,v):
        history = History()
        
        m1 = []
        for i, (word, tag) in enumerate(sentence[2:-1]):
#             if word in constants.IGNORE_WORDS:
#                 continue
            
            i += 2
            tm2 = sentence[i-2][1]
            tm1 = sentence[i-1][1]
            w = word
            wm1 = sentence[i-1][0]
            wp1 = sentence[i+1][0]
            history.set(tm2, tm1, wm1, w, wp1)
            
            denum = utils.calc_prob_denum(self.feat_manager, history, v)
            
            m2 = []
            for t in constants.TAGS: 
#                 if t in constants.IGNORE_TAGS:
#                     continue
                
                indices = self.feat_manager.calc_feature_vec(history, t)
                if not indices:
                    continue
                else:
                    curr = np.zeros_like(v)
                    curr[indices] = exp(sum(v[indices])) / denum
                    m2.append(curr)
            
            m1.append(sum(m2)) 
        
        return sum(m1)
    
    def calc_expected_counts(self, v):
        m = []
        for s in self.sentences:
            m.append(self.calc_expected_counts_aux(s, v))
        
        return np.array(sum(m))
    
    def loss_function_aux_func(self, sentence, v):
        history = History()
        
        outersum = np.zeros(len(sentence) - 3)
        for i, (word, tag) in enumerate(sentence[2:-1]):
#             if word in constants.IGNORE_WORDS:
#                 continue
            
            i += 2
            tm2 = sentence[i-2][1]
            tm1 = sentence[i-1][1]
            wm1 = sentence[i-1][0]
            wp1 = sentence[i+1][0]
            history.set(tm2, tm1, wm1, word, wp1)
            
            innersum = np.zeros(len(constants.TAGS))
            for j, tag in enumerate(constants.TAGS):
                indices = self.feat_manager.calc_feature_vec(history, tag)
                if not indices:
                    innersum[j] = 1
                else:
                    innersum[j] = exp(sum(v[indices]))
            
            outersum[i-2] = log(sum(innersum))   
        return sum(outersum)
    

def loss_function(v):
    print("Calculating L(v)...")
    start_time = time.process_time()
    
    global optimizer
    term1 = (optimizer.feat_metrix * v).sum()
    
    term2 = 0.0
    for s in optimizer.sentences:
        term2 += optimizer.loss_function_aux_func(s, v)
        
    term3 = (optimizer.lambda_param/2) * (LA.norm(v)**2)
    
    print("Done. Elapsed time:", time.process_time() - start_time)
    return -(term1 - term2 - term3)

def loss_function_der(v):
    print("Calculating L'(v)...")
    start_time = time.process_time()
    
    global optimizer
    term1 = optimizer.term1_of_loss_function_der
    term2 = optimizer.calc_expected_counts(v)
    term3 = optimizer.lambda_param * v
    
    der = np.zeros_like(v)
    der[:] = -(term1 - term2 - term3)
    
    print("Done. Elapsed time:", time.process_time() - start_time)
    return der 