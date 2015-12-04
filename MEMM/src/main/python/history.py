from _overlapped import NULL

class History(object):
    '''
    Represents a History
    '''
    
    def __init__(self):
        self.tm2 = None # t minus 2
        self.tm1 = None # t minus 1
        self.wm1 = None # w minus 1
        self.w = None   # w
        self.wp1 = None # w plus 1
    
    def set(self, tm2, tm1, wm1, w, wp1):
        self.tm2 = tm2
        self.tm1 = tm1
        self.wm1 = wm1
        self.w = w
        self.wp1 = wp1
        