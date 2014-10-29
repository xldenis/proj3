from math import exp

def sigmoid(x):
  return 1 / (1 + exp(-1*x))

def evaluate(weights, example):
  vec = example
  for mat in weights:
    vec = [ sigmoid(sum([a*b for a,b in zip(row,vec)])) for row in mat]
  return vec

def backprop(weights, example, label):
  outputs = []
  vec = example
  for mat in weights:
    vec = [ sigmoid(sum([a*b for a,b in zip(row,vec)])) for row in mat]
    outputs.append(vec)
  delta = [0] * len(weights)
  delta[len(weights) - 1] = [ o*(1-o)*(l-o) for l,o in zip(label, outputs[-1]) ]
  for idx in range(len(weights)-1)[::-1]:
    vec = outputs[idx]
    layer_update = [
      sum([a*b for a,b in zip(row, delta[idx+1])])
      for row in transpose(weights[idx+1])]
    delta[idx] = [o*(1-o)*u for o,u in zip(vec,layer_update)]

  for idx in range(len(weights)):
    update = sum([ a*b for a,b in zip(delta[idx], outputs[idx])])
    for rid in range(len(weights[idx])):
      weights[idx][rid] = [n + update for n in weights[idx][rid]]
  return weights 

def transpose(mat):
  return map(list, zip(*mat))

def init_weights(layers):
  mats = []
  for i in range(len(layers)-1):
    mat = []
    mats.append(mat)
    for j in range(layers[i+1]):
      mat.append([1 for k in range(layers[i])])
  return mats


def main():
  w = init_weights([8,3,8])
  for x in range(8):
    num = [0]*8
    num[x] = 1
    w = backprop(w, num, num)
  # print w
  print evaluate(w, [0,1,0,0,0,0,0,0])
if  __name__ =='__main__':main()
