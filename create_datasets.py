import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import os
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

import numpy.random

def one_vs_all(dataset = 'mnist', normal_class = 0, anom_size = 10, seed = None):
    if dataset == 'mnist':
        (x_train, y_train), (_, _) = mnist.load_data()
    if dataset == 'fmnist':
        (x_train, y_train), (_, _) = fashion_mnist.load_data()
    if dataset == 'cifar':
        (x_train, y_train), (_, _) = cifar10.load_data()
        y_train = y_train.reshape(y_train.shape[0])

    x_train = x_train.astype('float32') / 255.

    np.random.seed(seed=seed)

    normal = x_train[y_train == normal_class]
    id_anom = []
    for c in range(10):
        if c!=normal_class:
            id_c = np.random.choice(np.where(y_train == c)[0],anom_size,replace=False)
            id_anom = np.concatenate((id_anom,id_c))

    id_anom = id_anom.astype(int)
    anom = x_train[id_anom]
    data = np.concatenate((normal,anom))
    labels = np.array([1]*normal.shape[0]+[-1]*anom.shape[0])

    return data,labels,id_anom

def load_ODDS(filename):
    data_path = 'C:/Users/lucaf/Desktop/UNICAL/datasets ODDS'
    mat = loadmat(data_path+os.sep+filename)
    data = mat['X']
    labels = mat['y'].ravel()
    labels = labels.astype(int)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data,labels

def latents(d):
    latent_dims = [d]
    i=1
    while (d//(4**i))>3:
        latent_dims = latent_dims + [d//(4**i)]
        i = i+1
    latent_dims = latent_dims + [2]
    return latent_dims

def one_vs_many(dig,dig_an,anomalies_partition,DATASET='mnist',seed=None):

    if DATASET == 'mnist':
        (x_train, y_train), (_, _) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        y_test = y_test.copy()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    total_anomalies = np.sum(anomalies_partition)
    np.random.seed(seed=seed)
    x_train = x_train.astype('float32') / 255.
    #x_test = x_test.astype('float32') / 255.
    I_dig = np.where(y_train == dig)[0]
    x_training = x_train[I_dig]

    y_dig = [1] * x_training.shape[0]
    i = 0
    id_anomaly_train = np.array([])
    labels = np.array([])
    while i < len(dig_an):
        jj = np.where(y_train == dig_an[i])[0]
        id_estratti = np.array([])
        while id_estratti.shape[0] < anomalies_partition[i]:
            new_id = np.random.randint(jj.shape[0])
            if jj[new_id] not in id_estratti:
                id_estratti = np.append(id_estratti, jj[new_id])
        id_estratti = id_estratti.astype(int)

        xjj = x_train[id_estratti]
        labels = np.concatenate((labels, y_train[id_estratti]))
        id_anomaly_train = np.concatenate((id_anomaly_train, id_estratti))

        x_training = np.concatenate((x_training, xjj))
        y_training = np.array([1] * I_dig_mod + [-1] * total_anomalies)
        id_anomaly_train = id_anomaly_train.astype(int)

        i = i + 1
    labels_norm = [dig]*I_dig.shape[0]
    labels = np.concatenate((labels_norm,labels))
    labels = labels.astype(int)
    return x_training, y_training, id_anomaly_train, labels



def one_vs_many_modificato(x_train, y_train, k, purezza):

    x_training = x_train[y_train == 1]
    x_class_flattened = x_training.reshape(x_training.shape[0], -1)

    model = NearestNeighbors(n_neighbors=k)
    model.fit(x_class_flattened)

    # Calcola le distanze ai k vicini più vicini
    distances, indexes = model.kneighbors(x_class_flattened)

    # Calcola gli score di anomalia sommando le distanze
    anomaly_score = np.sum(distances, axis=1)

    # Salva gli score su un file
    with open('anomaly_scores.txt', 'a') as file:

        for score in anomaly_score:
            print(score, file=file)

    anomaly_thresh = np.percentile(anomaly_score, purezza)

    indices_below_threshold = np.where(anomaly_score <= anomaly_thresh)[0]
    indices_anomalies = y_train != 1
    x_training = x_training[indices_below_threshold]
    x_training = np.concatenate([x_training, x_train[indices_anomalies]], axis=0)
    y_training = np.ones(len(indices_below_threshold) + len(indices_anomalies))
    y_training[len(indices_below_threshold):len(y_training)] = -1

    return x_training, y_training

def create_dataset(dig,dig_an,anomalies_partition,DATASET='mnist',seed=None):
    np.random.seed(seed)
    x_train = None
    y_train = None
    if DATASET == 'mnist':
        (x_train, y_train), (_, _) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        y_test = y_test.copy()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    #x_test = x_test.astype('float32') / 255.
    num_anomalies = len(dig_an)
    ids_normal = np.where(y_train == dig)[0]
    total_normal = np.shape(ids_normal)[0]
    total_anomalies = np.sum(anomalies_partition)
    ids_train = np.zeros(total_normal + total_anomalies)
    ids_train[0:total_normal] = ids_normal[:]
    labels = np.int32(np.zeros(total_normal + total_anomalies))
    labels[0:total_normal] = dig
    x_training = np.zeros_like(x_train[0:total_normal + total_anomalies])
    #x_training = np.zeros((total_normal + total_anomalies, x_train.shape[1]))
    x_training[0:total_normal] = x_train[ids_normal]
    j = total_normal
    for i in range(num_anomalies):
        ids_anomalies_i = np.random.permutation(np.where(y_train == dig_an[i])[0])[0:anomalies_partition[i]]
        x_training[j:j + anomalies_partition[i]] = x_train[ids_anomalies_i]
        ids_train[j:j + anomalies_partition[i]] = ids_anomalies_i
        labels[j:j + anomalies_partition[i]] = dig_an[i]
        j = j + anomalies_partition[i]
    y_training = np.ones(total_normal + total_anomalies)
    y_training[total_normal:total_normal+total_anomalies] = -1
    return x_training, y_training, ids_train, labels

