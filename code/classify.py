import caffe
import numpy as np
import os



MODEL_FILE = './lenet.prototxt'
PRETRAINED = './lenet_iter_20000.caffemodel'

TRAINING_PATH = '/home/comp598/data/train_images'

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
# img = np.array([list(png.Reader(training_path+'/'+file[0]).read()[2])]).astype(np.uint8)
net.set_phase_test()

test = np.load('/dev/data_and_scripts/test_inputs.npy')
test = test.astype('float32')

outputs = []
for idx, val in enumerate(test):
  val = val.reshape(48,48,1)
  pred = net.predict([val])
  print idx
  outputs.append(pred[0].argmax())

print outputs