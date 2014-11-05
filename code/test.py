def kfold(k, classes, data):
  # data is tuple (text, id, label)
  # params is list of values to test
  k_size = int(round(len(data)/ k))
  groups = [data[i:i+ k_size] for i in range(0, len(data),k_size)]
  errors = [0] * k
  for i in range(0,k):
    print 'Fold %d' % i
    test = [idx[0] for idx in groups[i]]
    train = []
    for j in range(0,k):
      if j != i:
        train.append(groups[j])

    preds = yield (train, test)
    yield None
    for j in range(len(test)):
      if preds[j] != test[j][2]:
        errors[i] += 1

    print errors
  print float(sum(errors))/len(errors)
