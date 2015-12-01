import utils
import numpy as np

from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, lil_matrix
from math import log, exp
from history import History

optimizer = None
    
class Optimizer(object):
    '''
    Optimizes the parameters of MEMM for a given set of features.
    '''

    def __init__(self, parser, generator):
        self.lambda_param = 50.0
        self.n = len(parser.get_word_tag_array()) - 3
        self.m = generator.get_num_features()
        self.word_tag_array = parser.get_word_tag_array()
        self.generator = generator

        self.feat_metrix = self.clac_features_matrix()
        
        global optimizer
        optimizer = self
#         print(self.feat_metrix)
#         print(self.feat_metrix.get_shape())       
#         self.foo()
#         print(self.loss_function(np.zeros(self.m)))
#         print(self.n)
#         print(self.loss_function_der(np.zeros(self.m)))
    
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
            
        data = np.ones(col.size, dtype=np.int)
                
        return csr_matrix((data, (row, col)), shape=(self.n, self.m), dtype=np.int)
    
    def loss_function_aux_func(self, i, v):
        history = History()
        
        i += 2
        tm2 = self.word_tag_array[i-2][1]
        tm1 = self.word_tag_array[i-1][1]
        w = self.word_tag_array[i]
        wm1 = self.word_tag_array[i-1][0]
        wp1 = self.word_tag_array[i+1][0]
        history.set(tm2, tm1, wm1, w, wp1, i)
        
        res = np.zeros(len(utils.TAGS))
        for j, tag in enumerate(utils.TAGS):    
            feat_vec = np.array(self.generator.calc_feature_vec(history, tag))
            m = lil_matrix((1, self.m))            
            m[0, feat_vec] = 1
            res[j] = m.tocsr().dot(v)[0]
            
#         print(res)        
        return res
    
    def optimize(self, v0):
#         res = minimize(loss_function, v0, method='BFGS', jac=loss_function_der, options={'disp': True})
        res = minimize(loss_function, v0, method='L-BFGS-B', jac=loss_function_der, options={'disp': True})
        return res.x
        
    def foo(self):
        row = np.array([0, 0, 1, 2, 2, 2, 3])
        col = np.array([0, 2, 2, 0, 1, 2, 2])
        data = np.array([1, 2, 3, 4, 5, 6, 1])
        m = csr_matrix((data, (row, col)), shape=(4, 3)).toarray()
        print(m)
    
    def calc_prob_denum_aux(self, history, v):
        res = np.zeros(len(utils.TAGS))
        for i, tag in enumerate(utils.TAGS):    
            feat_vec = np.array(self.generator.calc_feature_vec(history, tag))
            m = lil_matrix((1, self.m))            
            m[0, feat_vec] = 1
            res[i] = m.tocsr().dot(v)[0]
                   
        return res
    
    def calc_prob(self, tag, history, v, denum):
        feat_vec = np.array(self.generator.calc_feature_vec(history, tag))
        m = lil_matrix((1, self.m), dtype=float)            
        m[0, feat_vec] = 1
        
        num = exp(m.tocsr().dot(v)[0])

        return num / denum
        
    def calc_expected_counts_aux(self, i ,v):
        history = History()
        
        i += 2
        tm2 = self.word_tag_array[i-2][1]
        tm1 = self.word_tag_array[i-1][1]
        w = self.word_tag_array[i]
        wm1 = self.word_tag_array[i-1][0]
        wp1 = self.word_tag_array[i+1][0]
        history.set(tm2, tm1, wm1, w, wp1, i)
        
        denum = sum(exp(p) for p in self.calc_prob_denum_aux(history, v))
        
        res = np.zeros(len(v))
        for tag in utils.TAGS:    
            feat_vec = np.array(self.generator.calc_feature_vec(history, tag))
            m = lil_matrix((1, self.m))            
            m[0, feat_vec] = 1
            prob = self.calc_prob(tag, history, v, denum)
            res += (m * prob).toarray()[0]
        
        return res
    
    def calc_expected_counts(self, v):
        res = np.zeros_like(v)
        for i in range(self.n):
            curr = self.calc_expected_counts_aux(i, v)
            res += curr
        
        return res
    

def loss_function(v):
    global optimizer
    term1 = (optimizer.feat_metrix * v).sum()
    term2 = sum(log(sum(exp(p) for p in optimizer.loss_function_aux_func(i, v))) for i in range(optimizer.n))
    term3 = (optimizer.lambda_param/2) * (LA.norm(v)**2)
    
    return -(term1 - term2 - term3)

def loss_function_der(v):
    global optimizer
    term1 = sum(optimizer.feat_metrix)
    term2 = optimizer.calc_expected_counts(v)
    term3 = optimizer.lambda_param * v
    
    der = np.zeros_like(v)
    der[:] = -(term1 - term2 - term3)
    
    return der 