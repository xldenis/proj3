from math import exp
from util import *
import random

def sigmoid(x):
  return 1 / (1 + exp(-1.0*x))

def evaluate(weights, example):
  vec = example
  for mat in weights:
    vec = [ sigmoid(sum([a*b for a,b in zip(row,vec)])) for row in mat]
  return vec

def backprop(weights, example, label):
  outputs = [example]
  vec = example
  for mat in weights:
    vec = [ sigmoid(sum([a*b for a,b in zip(row,vec)])) for row in mat]
    outputs.append(vec)
  delta = [0] * len(weights)
  delta[len(weights) - 1] = [ o*(1-o)*(l-o) for l,o in zip(label, outputs[-1]) ]
  for idx in range(len(weights)-1)[::-1]:
    vec = outputs[idx+1]
    layer_update = [
      sum([a*b for a,b in zip(row, delta[idx+1])])
      for row in transpose(weights[idx+1])]
    delta[idx] = [o*(1-o)*u for o,u in zip(vec,layer_update)]

  for idx in range(len(weights)):
    for rid in range(len(weights[idx])):
      weights[idx][rid] = [weights[idx][rid][n] + 0.8*outputs[idx][n]*delta[idx][rid] for n in range(len(weights[idx][rid]))]
  return weights 

def transpose(mat):
  return map(list, zip(*mat))

def init_weights(layers):
  mats = []
  for i in range(len(layers)-1):
    mat = []
    mats.append(mat)
    for j in range(layers[i+1]):
      mat.append([random.random() for k in range(layers[i])])
  return mats


def main():
  # training = load_training()
  # random.shuffle(training)
  # w = init_weights([48*48, 500, 10])
  i = 0
  w = init_weights([3,2,3])
  for i in xrange(0,50000):
    print i
    l = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    for lab in l:
      w = backprop(w, lab, lab)

  print "True label: %s and pred %s" % ([0,0,1], evaluate(w, [0,0,1]))
#   for ex in training[:1000]:
#     label = [0]*10
#     label[ex[2]] = 1
#     w = backprop(w, ex[0], label)
#     print "%s %s" %(label, i)
#     i += 1
#   for tes in training[:100]:
#     label = [0]*10
#     label[ex[2]] = 1
#     print "True label: %s and pred %s" % (label, evaluate(w, label))
if  __name__ =='__main__':main()
