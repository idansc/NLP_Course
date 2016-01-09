
class LabeledToken(object):
    '''
    Encapsulates a labeled token
    '''
    
    def __init__(self, idx, token, pos, head):
        self.idx = idx          # integer
        self.token = token      # string
        self.pos = pos          # string
        self.head = head        # integer
        self.in_between = None    # list of strings
