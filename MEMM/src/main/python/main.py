from features import buildFeatureVecFunc
from gd import GD
#import numpy as np
from features import get_q_func, unzip
from features import BASIC_FEATURES, FREQUENT_WORD_FEATURES, UNFREQUENT_WORD_FEATURES
from inference import inference
from collections import defaultdict
#import cProfile, pstats, StringIO

# def getTags():
#     f = open('sec2-21.wtag')
#     tags = set()
#     for i,line in enumerate(f):
#         if i >= 5000:
#             break
#         line = line.split()
#         for wt in line:
#             word, tag = wt.split('_')
#             tags.add(tag)
#     return tags


f = open('corpus\\sec2-21.wtag')
trainingSet = []
testSet = []
trainSetsize = 5000
testSetSize = 0
allTags = set()
wordTags = defaultdict(lambda:set())
for i,line in enumerate(f):
    line = line.split()
    #line = line[:-1]
    taggedSentence =[]
    for j,wt in enumerate(line):
        word, tag = wt.split('_')
        taggedSentence.append((word, tag))
        allTags.add(tag)
        wordTags[word].add(tag)
    if i < trainSetsize:
        trainingSet.append(taggedSentence)
        continue    
    if i > trainSetsize + testSetSize:
        break
    testSet.append(taggedSentence)        
        #trainingSet[(i,j)] = (word, tag)
#print trainingSet[0]
testSet = trainingSet[:2000]
print "Basic Training Features"
allTags.add('*') 
print 'len(trainingSet)', len(trainingSet)
print allTags
feature_templates = BASIC_FEATURES
f, featureVecSize = buildFeatureVecFunc(trainingSet, feature_templates)
#allTags.add('**')
gd = GD(f, trainingSet, allTags, featureVecSize)
print gd.getFeatureVecLen()
#pr=cProfile.Profile()
#pr.enable()
v = gd.fit(maxiter_p=10)
#pr.disable()
print v
for e in v:
    if float(e) > 1.0:
        print e
print 'len(v)', len(v)
from numpy import linalg as LA
print 'norm: ', str(LA.norm(v))
#testSet = trainingSet[:10]
s=0.0
#testSet=trainingSet[:5]
for taggedSentence in testSet:
    words,tags = unzip(taggedSentence)
    q_funk= get_q_func(f, v, allTags, words)
    prediction=inference(words, allTags, q_funk)
    hits = 0
    print words
    print tags
    print prediction
    for expected, tag in zip(tags, prediction):
        if expected == tag:
            hits+=1
    grade = float(hits)/len(tags)
    s+=grade
    print grade 
print s/len(testSet)
"""
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
"""
    

    
