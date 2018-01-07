"""
Author: Yikang Liao
Email: yliao1@andrew.cmu.edu
"""
import numpy as np


fname_train = 'digitstrain.txt'
fname_valid = 'digitsvalid.txt'
fname_test = 'digitstest.txt'


def load_data():
    X_train, y_train = load_from_file(fname_train)
    X_val, y_val = load_from_file(fname_valid)
    X_test, y_test = load_from_file(fname_test)
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def load_from_file(fname):
    data = np.loadtxt(fname, delimiter=',')
    #X = np.copy(data[:, 0:784])
    #Y = np.copy(data[:,784])
    X = data[:, 0:784]
    y = data[:,784]
    y = y.astype(int)
    return X, y


# def load_data():
#     data_train = load_from_file(fname_train)
#     data_valid = load_from_file(fname_valid)
#     data_test = load_from_file(fname_test)
#     return (data_train, data_valid, data_test)

# def load_from_file(fname):
#     data = np.loadtxt(fname, delimiter=',')
#     #X = np.copy(data[:, 0:784])
#     #Y = np.copy(data[:,784])
#     X = [np.reshape(x, (784, 1)) for x in data[:, 0:784]]
#     Y = [vectorize(int(i)) for i in data[:,784]]
#     return zip(X, Y)

# def vectorize(j):
#     #print(j)
#     e = np.zeros((10, 1))
#     e[j] = 1.0
#     return e

