import numpy as np
import pdb
import csv


def get_train_inputs(num_inputs=None):
    # Load all training inputs to a python list
    train_inputs = []
    with open('train_inputs.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
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


def make_folds(num_folds, set_size):
    all_inputs = get_train_inputs(set_size)
    all_outputs = get_train_outputs(set_size)
    fold_size = len(all_inputs) / num_folds
    for test_set_start_idx in range(0, len(all_inputs), fold_size):
        test_set_end_idx = test_set_start_idx + fold_size
        test_set_inputs = all_inputs[test_set_start_idx:test_set_end_idx]
        test_set_outputs = all_outputs[test_set_start_idx:test_set_end_idx]
        train_set_inputs = np.concatenate([all_inputs[:test_set_start_idx], all_inputs[test_set_end_idx:]])
        train_set_outputs = np.concatenate([all_outputs[:test_set_start_idx], all_outputs[test_set_end_idx:]])
        yield train_set_inputs, train_set_outputs, test_set_inputs, test_set_outputs


from sklearn.naive_bayes import GaussianNB
def get_classifier():
    return GaussianNB()

def cross_validate(num_folds, set_size):
    clf = get_classifier()
    for train_in, train_out, test_in, test_out in make_folds(num_folds, set_size):
        print("Train set size: %d, Test set size: %d" % (len(train_in), len(test_in)))
        clf.fit(train_in, train_out)
        predict_out = clf.predict(test_in)
        correct = [i for i in range(len(test_out)) if predict_out[i] == test_out[i]]
        accuracy = float(len(correct)) / len(test_in)
        print("Accuracy: %.3f" % accuracy)


if __name__ == '__main__':
    try:
        set_size = int(raw_input("Enter the total number of examples to use (leave blank to use all): "))
    except ValueError:
        set_size = None
    try:
        num_folds = int(raw_input("Enter the number of folds to use (leave blank to use 4): "))
    except ValueError:
        num_folds = 4
    cross_validate(num_folds, set_size)
