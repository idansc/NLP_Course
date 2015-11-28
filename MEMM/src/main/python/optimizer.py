import utils
import numpy as np

from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, lil_matrix
from math import log, exp
from history import History
    
class Optimizer(object):
    '''
    Optimizes the parameters of MEMM for a given set of features.
    '''

    def __init__(self, parser, generator):
        self.lambda_param = 0.5
        self.n = len(parser.get_word_tag_array()) - 3
        self.m = generator.get_num_features()
        self.word_tag_array = parser.get_word_tag_array()
        self.generator = generator

        self.feat_metrix = self.clac_features_matrix()
#         print(self.feat_metrix)
#         print(self.feat_metrix.get_shape())
        
        print(self.loss_function(np.empty(self.m)))
    
    def clac_features_matrix(self):
        history = History()
        col = np.array([], dtype=np.int)
        row = np.array([], dtype=np.int)

        for i, (word, tag) in enumerate(self.word_tag_array[2:-1]):
#             if word in [utils.START_SYMBOL, utils.END_SYMBOL, utils.DOT]:
#                 continue
            
            i += 2
            tm2 = self.word_tag_array[i-2][1]
            tm1 = self.word_tag_array[i-1][1]
            wm1 = self.word_tag_array[i-1][0]
            wp1 = self.word_tag_array[i+1][0]
            history.set(tm2, tm1, wm1, word, wp1, i)
            
            feat_vec = np.array(self.generator.calc_feature_vec(history, tag))
            col = np.concatenate((col, feat_vec))
            row = np.concatenate((row, np.full(feat_vec.size, i-2, dtype=np.int)))
            
        data = np.full(col.size, 1, dtype=np.int)
                
        return csr_matrix((data, (row, col)), shape=(self.n, self.m))
    
    def loss_function_aux_func(self, i, v):
        history = History()
        
        i += 2
        tm2 = self.word_tag_array[i-2][1]
        tm1 = self.word_tag_array[i-1][1]
        w = self.word_tag_array[i]
        wm1 = self.word_tag_array[i-1][0]
        wp1 = self.word_tag_array[i+1][0]
        history.set(tm2, tm1, wm1, w, wp1, i)
        
        res = []
        for tag in utils.TAGS:    
            feat_vec = np.array(self.generator.calc_feature_vec(history, tag))
            m = lil_matrix((1, self.m), dtype=np.int)
            m[0, feat_vec] = 1
            res.append(m.tocsr().dot(v)[0])
                
        return res
    
    def optimize(self):
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})
        print(res.x)
        
#     def foo(self):
#         n = len(word_tag_array)
#         m = n * (3 if self.extended_mode is True else 8)
#         F = lil_matrix(n, m)
#         print(n,m)
#         print(F)
#         pass

    def loss_function(self, v):
        term1 = (self.feat_metrix * v).sum()
        term2 = sum(log(sum(exp(p) for p in self.loss_function_aux_func(i, v))) for i in range(self.n))
        term3 = (self.lambda_param/2) * (LA.norm(v)**2)
        
        return term1 - term2 - term3
    
    def loss_function_der(self, v):
        pass

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der   