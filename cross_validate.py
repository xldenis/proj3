import numpy as np
import pdb
import csv
import preprocess


def get_train_inputs(in_file, num_inputs=None):
    # Load all training inputs to a python list
    train_inputs = []
    reader = csv.reader(in_file, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
        for dimension in train_input[1:]:
            train_input_no_id.append(float(dimension))
        train_inputs.append(np.asarray(train_input_no_id)) # Load each sample as a numpy array, which is appened to the python list
        if num_inputs and len(train_inputs) >= num_inputs:
            break
    return np.asarray(train_inputs)


def get_train_outputs(num_outputs=None):
    # Load all training ouputs to a python list
    train_outputs = []
    with open('train_outputs.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        for train_output in reader:  
            train_output_no_id = int(train_output[1])
            train_outputs.append(train_output_no_id)
            if num_outputs and len(train_outputs) >= num_outputs:
                break
    return np.asarray(train_outputs)


def make_folds(train_inputs, train_outputs, num_folds):
    fold_size = len(train_inputs) / num_folds
    for test_set_start_idx in range(0, len(train_inputs), fold_size):
        test_set_end_idx = test_set_start_idx + fold_size
        test_set_inputs = train_inputs[test_set_start_idx:test_set_end_idx]
        test_set_outputs = train_inputs[test_set_start_idx:test_set_end_idx]
        train_set_inputs = np.concatenate([train_inputs[:test_set_start_idx], train_inputs[test_set_end_idx:]])
        train_set_outputs = np.concatenate([train_outputs[:test_set_start_idx], train_outputs[test_set_end_idx:]])
        yield train_set_inputs, train_set_outputs, test_set_inputs, test_set_outputs


from sklearn.naive_bayes import GaussianNB
def get_classifier():
    return GaussianNB()

def cross_validate(inputs, outputs, num_folds):
    for train_in, train_out, test_in, test_out in make_folds(inputs, outputs, num_folds):
        print("Train set size: %d, Test set size: %d" % (len(train_in), len(test_in)))
        clf = get_classifier()
        clf.fit(train_in, train_out)
        predict_out = clf.predict(test_in)
        correct = [i for i in range(len(test_out)) if predict_out[i] == test_out[i]]
        accuracy = float(len(correct)) / len(test_in)
        print("Accuracy: %.3f" % accuracy)


if __name__ == '__main__':
    filename = raw_input("Enter the name of the train input file (leave blank to use train_inputs.csv): ")
    if filename == "":
        f = open("train_inputs.csv")
    else:
        f = open(filename)
    try:
        set_size = int(raw_input("Enter the total number of examples to use (leave blank to use all): "))
    except ValueError:
        set_size = None
    inputs = get_train_inputs(f, set_size)
    f.close()
    outputs = get_train_outputs(set_size)
    try:
        num_folds = int(raw_input("Enter the number of cross-validation folds to use (leave blank to use 4): "))
    except ValueError:
        num_folds = 4
    try:
        num_components = int(raw_input("Enter the number of features to keep (leave blank to skip pca): "))
        inputs = preprocess.pca(inputs, num_components)
    except ValueError:
        pass

    cross_validate(inputs, outputs, num_folds)
