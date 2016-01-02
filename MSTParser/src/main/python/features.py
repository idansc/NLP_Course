from abc import ABCMeta, abstractmethod
from collections import OrderedDict

class LocalFeatureTemplate(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self):
        self.features = OrderedDict()
    
    def add_local_feature(self, sentence, head, modifier):
        cnt = self.features.get(self.get_key(sentence, head, modifier), -1)
        if cnt == -1:
            self.features[self.get_key(sentence, head, modifier)] = 1
        else:
            self.features[self.get_key(sentence, head, modifier)] = cnt + 1
        
    def eval(self, sentence, head, modifier):
        return 1 if self.get_key(sentence, head, modifier) in self.features else 0
    
    def filter(self, threshold, idx):
        filtered_features = OrderedDict()
        for k, v in self.features.items():
            if v >= threshold:
                filtered_features[k] = idx
                idx += 1
        self.features = filtered_features
        return idx
        
    def get_feature_index(self, sentence, head, modifier):
        return self.features[self.get_key(sentence, head, modifier)]
    
    @abstractmethod
    def get_key(self, sentence, head, modifier):
        pass


class BaseLFT1(LocalFeatureTemplate):
    '''
    Base local feature template no. 1
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[head].pos)


class BaseLFT2(LocalFeatureTemplate):
    '''
    Base local feature template no. 2
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[head].token

class BaseLFT3(LocalFeatureTemplate):
    '''
    Base local feature template no. 3
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[head].pos


class BaseLFT4(LocalFeatureTemplate):
    '''
    Base local feature template no. 4
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[modifier].token, sentence[modifier].pos)


class BaseLFT5(LocalFeatureTemplate):
    '''
    Base local feature template no. 5
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[modifier].token


class BaseLFT6(LocalFeatureTemplate):
    '''
    Base local feature template no. 6
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[modifier].pos


class BaseLFT8(LocalFeatureTemplate):
    '''
    Base local feature template no. 8
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].pos, sentence[modifier].token, sentence[modifier].pos)


class BaseLFT10(LocalFeatureTemplate):
    '''
    Base local feature template no. 10
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[head].pos, sentence[modifier].pos)


class BaseLFT13(LocalFeatureTemplate):
    '''
    Base local feature template no. 13
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].pos, sentence[modifier].pos)


