from abc import ABCMeta, abstractmethod

class FeatureTemplate(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self):
        self.features = {}
    
    @abstractmethod
    def add_feature(self, history, tag, index):
        pass

    @abstractmethod
    def eval(self, history, tag):
        pass
    
    @abstractmethod
    def get_feature_index(self, history, tag):
        pass


class BaseFeatureTemplate1(FeatureTemplate):
    '''
    f100
    '''
    
    def add_feature(self, history, tag, index):
        self.features[(history.w, tag)] = index
        
    def eval(self, history, tag):
        return 1 if (history.w, tag) in self.features else 0
    
    def get_feature_index(self, history, tag):
        return self.features.get((history.w, tag), -1)


class BaseFeatureTemplate2(FeatureTemplate):
    '''
    f104
    '''
    
    def add_feature(self, history, tag, index):
        self.features[(history.tm1, tag)] = index
        
    def eval(self, history, tag):
        return 1 if (history.tm1, tag) in self.features else 0
    
    def get_feature_index(self, history, tag):
        return self.features.get((history.tm1, tag), -1)


class BaseFeatureTemplate3(FeatureTemplate):
    '''
    f103
    '''
    
    def add_feature(self, history, tag, index):
        self.features[(history.tm2, history.tm1, tag)] = index
        
    def eval(self, history, tag):
        return 1 if (history.tm2, history.tm1, tag) in self.features else 0
    
    def get_feature_index(self, history, tag):
        return self.features.get((history.tm2, history.tm1, tag), -1)
        