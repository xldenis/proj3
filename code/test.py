def kfold(k, classes, data):
  # data is tuple (text, id, label)
  # params is list of values to test
  k_size = int(round(len(data[0])/ k))
  groups = [(data[0][i:i + k_size],data[1][i:i + k_size],data[2][i:i + k_size]) for i in range(0, len(data[0]), k_size)]
  errors = [0] * k
  for i in range(0,k):
    print 'Fold %d' % i
    test = groups[i]
    t_data, t_ids, t_labs = [],[],[]
    for j in range(0,k):
      if j != i:
        t_data += (groups[j][0])
        t_ids += (groups[j][1])
        t_labs += (groups[j][2])

    preds = yield (t_data, t_ids, t_labs, test[0])
    yield None
    # for d in zip(*test):
    for j in range(len(zip(*test))):
      if preds[j] != zip(*test)[j][2]:
        errors[i] += 1
    print errors
  print float(sum(errors))/len(errors)/len(test[0])

def measure(c, classes, test, labels):
  print 'Testing'
  correct = 0
  both_wrong = 0
  one_right  = 0
  errors = defaultdict(lambda:0)
  test_length = len(test)
  for d in test[:test_length]:
    pred,scores = label(classes,*c,doc=d['abs'])
    if pred == labels[d['id']]:
      correct += 1
    else:
      errors[labels[d['id']]] += 1
  print errors #confusion
  print len(test)

def compare(c1,c2, classes, test, labels):
  print 'Testing'
  correct = [0,0]
  both_wrong = 0
  one_right  = 0
  errors = [defaultdict(lambda:0),defaultdict(lambda:0)]
  test_length = len(test)
  for d in test[:test_length]:
    pred = label(classes,*c1,doc=d['abs'])[0]
    if pred == labels[d['id']]:
      correct[0] += 1
    else:
      errors[0][labels[d['id']]] +=1
    pred2 = label(classes, *c2, doc=d['abs'])[0]
    if pred2 == labels[d['id']]:
      correct[1] += 1
    else:
      errors[1][labels[d['id']]] +=1

    if (pred2 == labels[d['id']]) and (pred == labels[d['id']]):
      one_right += 1
    if not (pred2 == labels[d['id']]) and not (pred == labels[d['id']]):
      both_wrong += 1
  print "Got %s, %s correct of %s" % (correct[0],correct[1], test_length) 
  print both_wrong
  print one_right 
  print errors[0]
  print errors[1]

def output(classes, c):
  test = load_test()
  for doc in test:
    print "\"%s\",\"%s\"" % (doc['id'], label(classes, *c, doc=doc['abs'])[0])
