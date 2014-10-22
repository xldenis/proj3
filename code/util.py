from itertools import chain
import os
import png
import csv

def load_training(training_path='../data/train_images'):
  train = []
  output = load_output()
  for file in zip(os.listdir(training_path), output):
    data = list(chain.from_iterable(png.Reader(training_path+"/"+file[0]).read()[2]))
    train.append((data, file[1][0],file[1][1]))
  return train

def load_output(label_path='../data/train_outputs.csv'):
  reader = csv.reader(open(label_path, 'r'),delimiter=',')
  label = []
  next(reader, None)
  for r in reader:
    label.append((int(r[0]),int(r[1])))
  return label

