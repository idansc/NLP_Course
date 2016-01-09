from abc import ABCMeta, abstractmethod
from collections import OrderedDict

class LocalFeatureTemplate(metaclass=ABCMeta):
    '''
    Defines common structure for all features
    '''
    
    def __init__(self):
        self.features = OrderedDict()
    
    def add_local_feature(self, sentence, head, modifier):
        cnt = self.features.get(self.get_key(sentence, head, modifier), 0)
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


class BasicLFT1(LocalFeatureTemplate):
    '''
    Basic local feature template no. 1
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[head].pos)


class BasicLFT2(LocalFeatureTemplate):
    '''
    Basic local feature template no. 2
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[head].token

class BasicLFT3(LocalFeatureTemplate):
    '''
    Basic local feature template no. 3
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[head].pos


class BasicLFT4(LocalFeatureTemplate):
    '''
    Basic local feature template no. 4
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[modifier].token, sentence[modifier].pos)


class BasicLFT5(LocalFeatureTemplate):
    '''
    Basic local feature template no. 5
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[modifier].token


class BasicLFT6(LocalFeatureTemplate):
    '''
    Basic local feature template no. 6
    '''
    
    def get_key(self, sentence, head, modifier):
        return sentence[modifier].pos


class BasicLFT8(LocalFeatureTemplate):
    '''
    Basic local feature template no. 8
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].pos, sentence[modifier].token, sentence[modifier].pos)


class BasicLFT10(LocalFeatureTemplate):
    '''
    Basic local feature template no. 10
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[head].pos, sentence[modifier].pos)


class BasicLFT13(LocalFeatureTemplate):
    '''
    Basic local feature template no. 13
    '''
    
    def get_key(self, sentence, head, modifier):
        return (sentence[head].pos, sentence[modifier].pos)


class ExtendedLFT1(LocalFeatureTemplate):
    '''
    Extended local feature template no. 7
    '''
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[head].pos, sentence[modifier].token, sentence[modifier].pos)


class ExtendedLFT2(LocalFeatureTemplate):
    '''
    Extended local feature template no. 9
    '''
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[modifier].token, sentence[modifier].pos)


class ExtendedLFT3(LocalFeatureTemplate):
    '''
    Extended local feature template no. 11
    '''
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[head].pos, sentence[modifier].token)


class ExtendedLFT4(LocalFeatureTemplate):
    '''
    Extended local feature template no. 12
    '''
    def get_key(self, sentence, head, modifier):
        return (sentence[head].token, sentence[modifier].token)


class ExtendedLFT5(LocalFeatureTemplate):
    '''
    Distance
    '''
    def get_key(self, sentence, head, modifier):
        return abs(head-modifier)

class ExtendedLFT6(LocalFeatureTemplate):
    '''
    p.pos, p.pos+1, c.pos-1, c.pos
    '''
    def get_key(self, sentence, head, modifier):
        p_pos_plus_1 = sentence[head+1].pos if head+1 < len(sentence) else None
        return (sentence[head].pos, p_pos_plus_1, sentence[modifier-1].pos, sentence[modifier].pos)

class ExtendedLFT7(LocalFeatureTemplate):
    '''
    p.pos-1, p.pos, c.pos-1, c.pos
    '''
    def get_key(self, sentence, head, modifier):
        return (sentence[head-1].pos, sentence[head].pos, sentence[modifier-1].pos, sentence[modifier].pos)

class ExtendedLFT8(LocalFeatureTemplate):
    '''
    p.pos, p.pos+1, c.pos, c.pos+1
    '''
    def get_key(self, sentence, head, modifier):
        p_pos_plus_1 = sentence[head+1].pos if head+1 < len(sentence) else None
        c_pos_plus_1 = sentence[modifier+1].pos if modifier+1 < len(sentence) else None
        return (sentence[head].pos, p_pos_plus_1, sentence[modifier].pos, c_pos_plus_1)

class ExtendedLFT9(LocalFeatureTemplate):
    '''
    p.pos-1, p.pos, c.pos, c.pos+1
    '''
    def get_key(self, sentence, head, modifier):
        c_pos_plus_1 = sentence[modifier+1].pos if modifier+1 < len(sentence) else None
        return (sentence[head-1].pos, sentence[head].pos, sentence[modifier].pos, c_pos_plus_1)
# 
# class InBetweenLFT1(LocalFeatureTemplate):
#     '''
#     p-pos, b-pos, c-pos
#     '''
#     def add_local_feature(self, sentence, head, modifier, between):
#         cnt = self.features.get(self.get_key(sentence, head, modifier), 0)
#         self.features[self.get_key(sentence, head, modifier)] = cnt + 1