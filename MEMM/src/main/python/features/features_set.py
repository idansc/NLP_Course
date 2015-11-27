from abc import ABCMeta, abstractmethod

class Feature(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self):
        self.samples = {}
    
    @abstractmethod
    def add_sample(self, history, tag):
        pass

    @abstractmethod
    def eval(self, history, tag):
        pass


class BaseFeature1(Feature):
    '''
    f100
    '''
    
    def add_sample(self, history, tag):
        self.samples.add((history.w, tag))
        
    def eval(self, history, tag):
        return 1 if (history.w, tag) in self.samples else 0


class BaseFeature2(Feature):
    '''
    f104
    '''
    
    def add_sample(self, history, tag):
        self.samples.add((history.tm1, tag))
        
    def eval(self, history, tag):
        return 1 if (history.tm1, tag) in self.samples else 0
        