from _overlapped import NULL

class History(object):
    '''
    A representation of a History
    '''
    def __init__(self):
        self.tm2 = NULL # t minus 2
        self.tm1 = NULL # t minus 1
        self.wm1 = NULL # w minus 1
        self.w = NULL   # w
        self.wp1 = NULL # w plus 1
        self.i = -1
    
    def set(self, tm2, tm1, wm1, w, wp1, i):
        self.tm2 = tm2
        self.tm1 = tm1
        self.wm1 = wm1
        self.w = w
        self.wp1 = wp1
        self.i = i
        