from abc import ABCMeta, abstractmethod
from collections import OrderedDict

class FeatureTemplate(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self):
        self.features = OrderedDict()
    
    def add_feature(self, sentence, edge):
        cnt = self.features.get(self.get_key(sentence, edge), -1)
        if cnt == -1:
            self.features[self.get_key(sentence, edge)] = 1
        else:
            self.features[self.get_key(sentence, edge)] = cnt + 1
        
    def eval(self, sentence, edge):
        return 1 if self.get_key(sentence, edge) in self.features else 0
    
    def filter(self, threshold, idx):
        filtered_features = OrderedDict()
        for k, v in self.features.items():
            if v >= threshold:
                filtered_features[k] = idx
                idx += 1
        self.features = filtered_features
        return idx
        
    def get_feature_index(self, sentence, edge):
        return self.features[self.get_key(sentence, edge)]
    
    @abstractmethod
    def get_key(self, sentence, edge):
        pass


class BaseFeatureTemplate1(FeatureTemplate):
    '''
    Feature no. 1
    '''
    
    def get_key(self, sentence, edge):
        return (sentence[edge[0]].token, sentence[edge[0]].pos)

