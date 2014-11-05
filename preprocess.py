import cross_validate
import sklearn.decomposition
import csv

def pca(inputs, num_components):
    pca = sklearn.decomposition.PCA(n_components=num_components)
    pca.fit(inputs)
    return pca.transform(inputs)

if __name__ == '__main__':
    try:
        set_size = int(raw_input("Enter the total number of examples to use (leave blank to use all): "))
    except ValueError:
        set_size = None
    train_inputs = cross_validate.get_train_inputs(10000)
    try:
        num_components = int(raw_input("Enter the number of features to keep (leave blank to use 1500): "))
    except ValueError:
        num_components = 1500
    transformed = transform(train_inputs, num_components)
    with open('transformed.csv', 'wb') as csvfile:
        a = csv.writer(csvfile, delimiter=',')
        for i, train_input in enumerate(transformed_inputs, start=1):
            row = [i]
            row.extend(train_input)
            a.writerow(row)
