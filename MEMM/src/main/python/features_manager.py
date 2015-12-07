from features import *
from history import History

class FeaturesManager(object):
    '''
    Generates set of features out of a given training data
    '''

    def __init__(self, parser, feat_threshold, use_advanced_features):
        self.feature_templates = [BaseFeatureTemplate1(), BaseFeatureTemplate2(), BaseFeatureTemplate3()]
        if use_advanced_features == True:
            self.addAdvancedFeatureTemplates(parser, feat_threshold)
        history = History()
        for s in parser.get_sentences():  
            for i,(word,tag) in enumerate(s[2:-1]):
#                 if word in [constants.START_SYMBOL, constants.END_SYMBOL, constants.DOT]:
#                     continue
                
                i += 2
                tm2 = s[i-2][1]
                tm1 = s[i-1][1]
                wm1 = s[i-1][0]
                wp1 = s[i+1][0]
                history.set(tm2, tm1, wm1, word, wp1)
                for template in self.feature_templates:
                    template.add_feature(history, tag)
        
        idx = 0
        for t in self.feature_templates:
            idx = t.filter(feat_threshold, idx)        
    
    def calc_feature_vec(self, history, tag):
        return [t.get_feature_index(history, tag) for t in self.feature_templates if t.eval(history, tag) == 1]
    
    def get_num_features(self):
        return sum(len(t.features) for t in self.feature_templates)
    
    def addAdvancedFeatureTemplates(self, parser, feat_threshold):
        self.feature_templates += [
                AdvancedFeatureTemplate1(parser.get_suffixes(1, feat_threshold), 1),
                AdvancedFeatureTemplate1(parser.get_suffixes(2, feat_threshold), 2),
                AdvancedFeatureTemplate1(parser.get_suffixes(3, feat_threshold), 3),
                AdvancedFeatureTemplate1(parser.get_suffixes(4, feat_threshold), 4),
                AdvancedFeatureTemplate2(parser.get_prefixes(1, feat_threshold), 1),
                AdvancedFeatureTemplate2(parser.get_prefixes(2, feat_threshold), 2),
                AdvancedFeatureTemplate2(parser.get_prefixes(3, feat_threshold), 3),
                AdvancedFeatureTemplate2(parser.get_prefixes(4, feat_threshold), 4),
                AdvancedFeatureTemplate3(),
                AdvancedFeatureTemplate4(),
                AdvancedFeatureTemplate5()
            ]
        pass