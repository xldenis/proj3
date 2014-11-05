import caffe
from caffe.proto import caffe_pb2
import lmdb
from itertools import chain
import numpy as np
import os
import png
import csv
import array
import skimage
import random

def load_output(label_path='data/train_outputs.csv'):
  reader = csv.reader(open(label_path, 'r'),delimiter=',')
  label = []
  next(reader, None)
  for r in reader:
    label.append((int(r[0]),int(r[1])))
  return label

def load_training(training_path='data/train_images'):
  train = []
  output = load_output()
  for file in zip(os.listdir(training_path), output):
    img = np.array([list(png.Reader(training_path+'/'+file[0]).read()[2])]).astype(np.uint8)
    # img = skimage.img_as_float(img).astype(np.float32)
    # data = list(chain.from_iterable(caffe.io.load_image(training_path+"/"+file[0], color=False)))
    train.append((img, file[1][0],file[1][1]))
  return train

training = load_training()
# random.shuffle(training)

env = lmdb.open('comp_mnist', create='True', map_size=1073741824)

training2 = training[:25000]
with env.begin(write=True) as txn:
  for data in training2:
    datum = caffe.io.array_to_datum(data[0],data[2])
    txn.put(str(data[1]), datum.SerializeToString())

env.close()

env = lmdb.open('comp_mnist_test', create='True', map_size=1073741824)

test = training[25000:]
with env.begin(write=True) as txn:
  for data in test:
    datum = caffe.io.array_to_datum(data[0],data[2])
    txn.put(str(data[1]), datum.SerializeToString())

env.close()