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

env = lmdb.open('comp_mnist', create='True')
txn = env.begin(write=True)

for data in training:
  datum = caffe.io.array_to_datum(data[0],data[2])
  txn.put(str(data[1]), datum.SerializeToString())
  
txn.commit()