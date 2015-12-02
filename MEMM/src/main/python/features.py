from abc import ABCMeta, abstractmethod

class FeatureTemplate(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self):
        self.features = {}
    
    def add_feature(self, history, tag):
        cnt = self.features.get(self.get_key(history, tag), -1)
        if cnt == -1:
            self.features[self.get_key(history, tag)] = 1
        else:
            self.features[self.get_key(history, tag)] = cnt + 1
        
    def eval(self, history, tag):
        return 1 if self.get_key(history, tag) in self.features else 0
    
    def filter(self, threshold, idx):
        filtered_features = {}
        for k, v in self.features.items():
            if v >= threshold:
                filtered_features[k] = idx
                idx += 1
        self.features = filtered_features
        return idx
        
    def get_feature_index(self, history, tag):
        return self.features[self.get_key(history, tag)]
    
    @abstractmethod
    def get_key(self, history, tag):
        pass


class BaseFeatureTemplate1(FeatureTemplate):
    '''
    f100
    '''
    
    def get_key(self, history, tag):
        return (history.w, tag)

class BaseFeatureTemplate2(FeatureTemplate):
    '''
    f104
    '''
    
    def get_key(self, history, tag):
        return (history.tm1, tag)


class BaseFeatureTemplate3(FeatureTemplate):
    '''
    f103
    '''
    
    def get_key(self, history, tag):
        return (history.tm2, history.tm1, tag)
        