import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from history import History
    
class Optimizer(object):
    '''
    Optimizes the parameters of MEMM for a given set of features.
    '''

    def __init__(self, parser, generator):
        word_tag_array = parser.get_word_tag_array()
        history = History()
        col = np.array([], dtype=np.int)
        row = np.array([], dtype=np.int)

        for i,(word,tag) in enumerate(word_tag_array[2:-1]):
            i += 2
            tm2 = word_tag_array[i-2][1]
            tm1 = word_tag_array[i-1][1]
            wm1 = word_tag_array[i-1][0]
            wp1 = word_tag_array[i+1][0]
            history.set(tm2, tm1, wm1, word, wp1, i)
            
            feat_vec = np.array(generator.calc_feature_vec(history, tag))
            col = np.concatenate((col, feat_vec))
            row = np.concatenate((row, np.full(feat_vec.size, i-2, dtype=np.int)))
            
        data = np.full(col.size, 1, dtype=np.int)
#         print(row, row.size)
#         print(col, col.size)
#         print(data, data.size)
        
        feat_metrix = coo_matrix((data, (row, col)))
        
#         print(feat_metrix)
#         print(feat_metrix.get_shape())

    
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