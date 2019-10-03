## Dataloader for MNIST

import _pickle as cPickle
import gzip
import numpy as np

def load():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)

def wrapper():
    train, valid, test = load()
    for x in train[0]:
        training_inputs = [np.reshape(x, (784, 1))]
    for y in train[1]:
        training_results = [vectorize(y)]
    training_data = list(zip(training_inputs, training_results))

    for x in valid[0]:
        validation_inputs = [np.reshape(x, (784, 1))]
    validation_data = list(zip(validation_inputs, valid[1]))

    for x in test[0]:
        test_inputs = [np.reshape(x, (784, 1))]
    test_data = list(zip(test_inputs, test[1]))
    return (training_data, validation_data, test_data)

def vectorize(idx):
    e = np.zeros((10, 1))
    e[idx] = 1
    return e