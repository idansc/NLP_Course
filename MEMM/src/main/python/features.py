from abc import ABCMeta, abstractmethod

class FeatureTemplate(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self, feat_threshold):
        self.features = {}
        self.threshold = feat_threshold
    
    def add_feature(self, history, tag, index):
        self.features[self.get_key(history, tag)] = [index, 1]
        
    def eval(self, history, tag):
        res = self.features.get(self.get_key(history, tag), -1)
        return 1 if res != -1 and res[1] >= self.threshold else 0
    
    def get_feature_index(self, history, tag):
        res = self.features.get(self.get_key(history, tag), -1)
        return -1 if res == -1 else res[0]
    
    def inc_count(self, history, tag):
        res = self.features[self.get_key(history, tag)]
        res[1] += 1
        self.features[self.get_key(history, tag)] = res
    
    @abstractmethod
    def get_key(self, history, tag):
        pass


class BaseFeatureTemplate1(FeatureTemplate):
    '''
    f100
    '''
    def __init__(self, feat_threshold):
        super().__init__(feat_threshold)
    
    def get_key(self, history, tag):
        return (history.w, tag)

class BaseFeatureTemplate2(FeatureTemplate):
    '''
    f104
    '''
    
    def __init__(self, feat_threshold):
        super().__init__(feat_threshold)
    
    def get_key(self, history, tag):
        return (history.tm1, tag)


class BaseFeatureTemplate3(FeatureTemplate):
    '''
    f103
    '''
    
    def __init__(self, feat_threshold):
        super().__init__(feat_threshold)
    
    def get_key(self, history, tag):
        return (history.tm2, history.tm1, tag)
        