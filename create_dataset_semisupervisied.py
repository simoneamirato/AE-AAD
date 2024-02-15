from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10
import numpy as np
from scipy.stats import randint
import os
from PIL import Image

def one_vs_many(dig,dig_an,anomalies_partition,DATASET='mnist',seed=None):
    if DATASET == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        y_test = y_test.copy()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    total_anomalies = np.sum(anomalies_partition)

    np.random.seed(seed=seed)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    I_dig = np.where(y_train == dig)[0]
    x_training = x_train[I_dig]
    y_dig = [0] * x_training.shape[0]
    i = 0
    while i < len(dig_an):
        jj = np.where(y_train == dig_an[i])[0]
        id_estratti = np.array([])
        id_anomaly_train = np.array([])
        while id_estratti.shape[0] < anomalies_partition[i]:
            new_id = np.random.randint(jj.shape[0])
            if new_id not in id_estratti:
                id_estratti = np.append(id_estratti, new_id)
        id_estratti = id_estratti.astype(int)
        xjj = x_train[jj[id_estratti]]
        id_anomaly_train = np.concatenate((id_anomaly_train, id_estratti))
        x_training = np.concatenate((x_training, xjj))
        y_training = np.array([0] * I_dig.shape[0] + [1] * total_anomalies)
        id_anomaly_train = id_anomaly_train.astype(int)
        i = i + 1

    y_test[y_test == dig] = 10
    y_test[y_test != 10] = 1
    y_test[y_test == 10] = 0
    return x_training, y_training, x_test, y_test


def one_vs_all(dig,num_anomalies,DATASET = 'mnist',seed=None):
    if DATASET == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(y_train.shape[0])
        y_test = y_test.reshape(y_test.shape[0])

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_tr_norm = x_train[y_train==dig]
    x_tr_tot_anom = x_train[y_train != dig]
    tot_anom = x_tr_tot_anom.shape[0]

    np.random.seed(seed=seed)

    extracted = np.random.choice(tot_anom, num_anomalies,replace=False)
    x_tr_anom = x_tr_tot_anom[extracted]

    x_training = np.concatenate((x_tr_norm,x_tr_anom))
    y_training = np.array([0]*x_tr_norm.shape[0]+[1]*num_anomalies)

    y_test = np.copy(y_test)
    y_test[y_test == dig] = 10
    y_test[y_test != 10] = 1
    y_test[y_test == 10] = 0
    return x_training, y_training, x_test, y_test


def one_vs_all_FRUIT(normal_class,num_anomalies,train_perc=0.9,seed=None):
    np.random.seed(seed=seed)
    classes = ['Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'Muskmelon', 'Orange',
               'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes']

    size = (120, 160, 3)
    sizePIL = (160, 120)

    num_class = len(os.listdir('FRUTTA/'+normal_class+'/')) #total number of normal items

    train_norm = int(num_class*train_perc)
    test_norm = num_class-train_norm

    x_train_norm = np.zeros((train_norm,).__add__(size))
    x_test_norm = np.zeros((test_norm,).__add__(size))

    for i in range(num_class):
        filename = normal_class + str(i+1) + '.png'
        im_frame = Image.open('FRUTTA/' + normal_class + '/' + filename)
        if i < train_norm:
            x_train_norm[i] = np.array(im_frame.resize(sizePIL))
        else:
            x_test_norm[train_norm - i] = np.array(im_frame.resize(sizePIL))

    classes_an = classes.copy()
    classes_an.remove(normal_class)

    anom_partition = np.random.multinomial(num_anomalies, [1/14.]*14)
    x_train_anom = np.zeros((num_anomalies,).__add__(size))

    x_test_anom = np.zeros((0,).__add__(size))

    at_index = 0

    for j in range(len(classes_an)):
        cl = classes_an[j]
        num_cl = len(os.listdir('FRUTTA/' + cl + '/'))
        train_cl = int(num_cl * train_perc)
        test_cl = num_cl - train_cl

        if anom_partition[j] != 0:
            extracted = np.random.choice(train_cl, anom_partition[j], replace=False)
            for k in range(anom_partition[j]):
                filename = cl + str(extracted[k]) + '.png'
                im_frame = Image.open('FRUTTA/' + cl + '/' + filename)
                x_train_anom[at_index] = np.array(im_frame.resize(sizePIL))
                at_index = at_index + 1

        x_test_cl = np.zeros((test_cl,).__add__(size))
        for i in range(test_cl):
            filename = cl + str(train_cl + i + 1) + '.png'
            im_frame = Image.open('FRUTTA/' + cl + '/' + filename)
            x_test_cl[i] = np.array(im_frame.resize(sizePIL))

        x_test_anom = np.concatenate((x_test_anom, x_test_cl))

    x_training = np.concatenate((x_train_norm, x_train_anom))
    y_training = np.array([0]*x_train_norm.shape[0]+[1]*x_train_anom.shape[0])
    x_test = np.concatenate((x_test_norm, x_test_anom))
    y_test = np.array([0]*x_test_norm.shape[0]+[1]*x_test_anom.shape[0])

    x_training = x_training/255.
    x_test = x_test/255.

    return x_training, y_training, x_test, y_test





















def square(dig,perc_anom_train,perc_anom_test,size=5,intensity = 'rand',DATASET = 'mnist'):
    if DATASET == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.


    width,height = x_train.shape[1:3]
    edge = size//2

    num_anomalies_train = int(perc_anom_train*(np.where(y_train==dig)[0].size))


    id_an_train = np.random.choice(np.where(y_train==dig)[0],num_anomalies_train,replace=False)
    GT_train = np.zeros((x_train.shape[0],width,height))


    for id in id_an_train:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)
        if intensity == 'rand':
            intens = np.random.randint(0,255,3)/255.
        if len(x_train.shape) == 4:
            x_train[id,center_x-edge:center_x+edge+1,center_y-edge:center_y+edge+1,0] = intens[0]
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 1] = intens[1]
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 2] = intens[2]
        else:
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = intens[0]

        GT_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1

    num_anomalies_test = int(perc_anom_test*(np.where(y_test==dig)[0].size))

    id_an_test = np.random.choice(np.where(y_test == dig)[0], num_anomalies_test,replace=False)
    GT_test = np.zeros((x_test.shape[0], width, height))

    for id in id_an_test:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)

        if intensity == 'rand':
            intens = np.random.randint(0, 255, 3) / 255.

        if len(x_test.shape) == 4:
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 0] = intens[0]
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 1] = intens[1]
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 2] = intens[2]
        else:
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = intens[0]

        GT_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1
    id_train = np.where(y_train == dig)
    id_test = np.where(y_test == dig)



    X_train = x_train[id_train]
    X_test = x_test[id_test]
    GT_train = GT_train[id_train]
    GT_test = GT_test[id_test]
    y_train[id_train] = 0
    y_train[id_an_train] = 1
    Y_train = y_train[id_train]
    y_test[id_test] = 0
    y_test[id_an_test] = 1
    Y_test = y_test[id_test]


    if len(X_test.shape) < 4:
        X_train = X_train.reshape(X_train.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

    X_train = np.swapaxes(X_train, 2, 3)
    X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.swapaxes(X_test, 2, 3)
    X_test = np.swapaxes(X_test, 1, 2)

    return X_train,Y_train,X_test,Y_test,GT_train,GT_test




def meta_aad_odds(test_dataset=None, n_train_datasets=4, seed=None):

    datasets_names = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
                      'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19',
                      'd20', 'd21', 'd22', 'd23', 'd24']

    datasets_names.remove(test_dataset)

    np.random.seed(seed=seed)

    a = np.random.choice(datasets_names, n_train_datasets, replace=False)

    return list(a)



