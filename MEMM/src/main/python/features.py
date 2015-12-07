from abc import ABCMeta, abstractmethod
from collections import OrderedDict

class FeatureTemplate(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self):
        self.features = OrderedDict()
    
    def add_feature(self, history, tag):
        cnt = self.features.get(self.get_key(history, tag), -1)
        if cnt == -1:
            self.features[self.get_key(history, tag)] = 1
        else:
            self.features[self.get_key(history, tag)] = cnt + 1
        
    def eval(self, history, tag):
        return 1 if self.get_key(history, tag) in self.features else 0
    
    def filter(self, threshold, idx):
        filtered_features = OrderedDict()
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
#         return (history.w.lower(), tag)
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

class AdvancedFeatureTemplate1(FeatureTemplate):
    '''
    f101. Suffixes of length n
    '''
    
    def __init__(self, suffixes, n):
        super().__init__()
        self.suffixes = suffixes
        self.n = n
    
    def get_key(self, history, tag):
        return (tag, history.w[-self.n:])
    
    def eval(self, history, tag):
        return 1 if (self.get_key(history, tag) in self.features) and (history.w[-self.n:] in self.suffixes) else 0
    
class AdvancedFeatureTemplate2(FeatureTemplate):
    '''
    f102. Prefixes of length n
    '''
    
    def __init__(self, prefixes, n):
        super().__init__()
        self.prefixes = prefixes
        self.n = n
         
    def get_key(self, history, tag):
#         return (tag, history.w[:self.n].lower())
        return (tag, history.w[:self.n])
     
    def eval(self, history, tag):
#         return 1 if (self.get_key(history, tag) in self.features) and (history.w[:self.n].lower() in self.prefixes) else 0
        return 1 if (self.get_key(history, tag) in self.features) and (history.w[:self.n] in self.prefixes) else 0
    
class AdvancedFeatureTemplate3(FeatureTemplate):
    '''
    f105
    '''
    
    def get_key(self, history, tag):
        return tag

class AdvancedFeatureTemplate4(FeatureTemplate):
    '''
    f106
    '''
     
    def get_key(self, history, tag):
        return (history.wm1, tag)

class AdvancedFeatureTemplate5(FeatureTemplate):
    '''
    f107
    '''
     
    def get_key(self, history, tag):
        return (history.wp1, tag)

class AdvancedFeatureTemplate6(FeatureTemplate):
    '''
    w_i contains a number and t_i = T
    '''
     
    def get_key(self, history, tag):
        return (tag, history.w.lower())
     
    def eval(self, history, tag):
        return 1 if (self.get_key(history, tag) in self.features) and (any(char.isdigit() for char in history.w)) else 0
    
class AdvancedFeatureTemplate7(FeatureTemplate):
    '''
    w_i contains an uppercase character and t_i = T
    '''
     
    def get_key(self, history, tag):
        return (tag, history.w.lower())
     
    def eval(self, history, tag):
        return 1 if (self.get_key(history, tag) in self.features) and (any(char.isupper() for char in history.w)) else 0       