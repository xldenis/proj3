import numpy as np
import random
from math import exp
from util import *

def quad_error(el):
  return el*(1-el)

def sigmoid(x):
  return 1 / (1 + exp(-1.0*x))

def extend(vec):
  return np.vstack((vec, np.array([[1]])))

def mat_backprop(weights, x, t):
  gamma = 0.8
  vec_sigmoid = np.vectorize(sigmoid)
  vfunc = np.vectorize(quad_error)

  outputs = [x]

  o = x
  for mat in weights:
    extended_o = extend(o)
    o = vec_sigmoid(extended_o.T.dot(mat)).T
    outputs.append(o)

  e = o - t

  diags = []

  for oi in outputs[1:]:
    diags.append(np.diagflat(vfunc(oi)))

  deltas = [0]*len(weights)
  deltas[-1] = diags[-1].dot(e)
  for idx in range(len(weights)-1)[::-1]:
    deltas[idx] = diags[idx].dot(weights[idx+1][:-1]).dot(deltas[idx+1])

  wnew = []
  for i in xrange(len(weights)):
    z = deltas[i].dot(extend(outputs[i]).T)
    wnew.append(weights[i] - gamma*z.T)

  return wnew

def evaluate(weights, o):
  vec_sigmoid = np.vectorize(sigmoid)
  for mat in weights:
    extended_o = extend(o)
    o = vec_sigmoid(extended_o.T.dot(mat)).T
  return o
def init_weights(layers):
  mats = []
  for i in range(len(layers)-1):
    mat = []
    mats.append(mat)
    for j in range(layers[i]+1):
      mat.append([random.random() for k in range(layers[i+1])])
  true_mats = []
  for m in mats:
    true_mats.append(np.array(m))
  return true_mats

training = load_training()
random.shuffle(training)
w = init_weights([48*48, 500, 10])
training2 = []
for t in training:
  l = [0] * 10
  l[t[2]] = 1
  training2.append((np.array([t[0]]).T, np.array([l]).T))
print training2[0][1]
for i in xrange(0,1):
  print i
  j = 0
  for x in training2:
    j += 1
    if (j % 1000) == 0:
      print j
    w = mat_backprop(w, x[0], x[1])

    