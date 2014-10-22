from math import log, sqrt
from collections import defaultdict
from util import *
from test import *
from nltk.stem import PorterStemmer
import random
import re
import itertools

def count(data, vocab):
  counts = {}
  for doc in data:
    counts[doc[1]] = defaultdict(lambda:0)
    for px in doc[0]:
      counts[doc[1]][px] += 1
  return counts

def train(classes, vocab, training_data, features):
  d = len(training_data)
  probs = {} #probs[c][w] = P(w|c)
  prior = {}

  for c in classes:
    print 'Training %s' % c
    ids_in_class = [x[1] for x in training_data if x[2] == c]
    probs[c] = defaultdict(lambda:1.0 / len(vocab))
    prior[c] = len(ids_in_class) / float(d)
    totalCount = 0
    sum_feat = defaultdict(lambda: 0)
    
    for idx in ids_in_class:
      for feat in features[idx].keys():
        totalCount += features[idx][feat]
        sum_feat[feat] += features[idx][feat]

    for feat in vocab:
      top = sum_feat[feat]
      if top > 0:
        probs[c][w] = (top + 1) / float(totalCount -top + d)
  return vocab, prior, probs

def label(classes, vocab, prior, probs, doc):
  score = {}
  for c in classes:
    score[c] = log(prior[c])
    for t in words:
      score[c] += log(probs[c][t])

  return max(score, key=score.get), score

def main(): 
  classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  training = load_training()
  random.shuffle(training)

  vocab = range(0,256)
  print training[0]
  features = count(training, vocab)
  classifier = train(classes, vocab, training, features)
if  __name__ =='__main__':main()
