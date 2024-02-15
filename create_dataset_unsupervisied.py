from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10
import numpy as np
import os


def one_vs_all(dig,size):
    # Crea un dataset composto da tutti le cifre "dig" di mnist e da size elementi di ogni altra cifra
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))

    I_dig = np.where(y_train == dig)[0]
    x_dig = x_train[I_dig]
    y_dig = [0] * x_dig.shape[0]
    x_training = x_dig
    id_anomaly_train = np.array([])
    for i in range(10):

        if (i != dig):
            jj = np.where(y_train == i)[0]
            id_estratti = np.array([])
            while id_estratti.shape[0] < size:
                new_id = np.random.randint(jj.shape[0])
                if new_id not in id_estratti:
                    id_estratti = np.append(id_estratti,new_id)
            id_estratti = id_estratti.astype(int)
            #xjj = x_train[jj[0:size]]
            xjj = x_train[jj[id_estratti]]
            id_anomaly_train = np.concatenate((id_anomaly_train,id_estratti))
            x_training = np.concatenate((x_training, xjj))

    y_nodig = [1] * 9 * size
    y_training = np.concatenate((y_dig, y_nodig))
    id_anomaly_train = id_anomaly_train.astype(int)
    return x_training,y_training,id_anomaly_train



def one_vs_all_fashion(dig,size):
    # Crea un dataset composto da tutti le cifre "dig" di mnist e da size elementi di ogni altra cifra
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))

    I_dig = np.where(y_train == dig)
    x_dig = x_train[I_dig]
    y_dig = [0] * x_dig.shape[0]
    x_training = x_dig
    id_anomaly = np.array([])
    for i in range(10):

        if (i != dig):
            jj = np.where(y_train == i)[0]
            id_estratti = np.array([])
            while id_estratti.shape[0] < size:
                new_id = np.random.randint(jj.shape[0])
                if new_id not in id_estratti:
                    id_estratti = np.append(id_estratti,new_id)
            id_estratti = id_estratti.astype(int)
            #xjj = x_train[jj[0:size]]
            xjj = x_train[jj[id_estratti]]
            id_anomaly = np.concatenate((id_anomaly,id_estratti))
            x_training = np.concatenate((x_training, xjj))

    y_nodig = [1] * 9 * size
    y_training = np.concatenate((y_dig, y_nodig))
    id_anomaly = id_anomaly.astype(int)
    return x_training,y_training,id_anomaly

def one_vs_all_cifar(dig,size):
    # Crea un dataset composto da tutti le cifre "dig" di mnist e da size elementi di ogni altra cifra
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    #x_train = x_train.reshape(x_train.shape + (1,))

    I_dig = np.where(y_train == dig)[0]
    x_dig = x_train[I_dig]
    y_dig = [0] * x_dig.shape[0]
    x_training = x_dig
    id_anomaly = np.array([])
    for i in range(10):

        if (i != dig):
            jj = np.where(y_train == i)[0]
            id_estratti = np.array([])
            while id_estratti.shape[0] < size:
                new_id = np.random.randint(jj.shape[0])
                if new_id not in id_estratti:
                    id_estratti = np.append(id_estratti,new_id)
            id_estratti = id_estratti.astype(int)
            #xjj = x_train[jj[0:size]]
            xjj = x_train[jj[id_estratti]]
            id_anomaly = np.concatenate((id_anomaly,id_estratti))
            x_training = np.concatenate((x_training, xjj))

    y_nodig = [1] * 9 * size
    y_training = np.concatenate((y_dig, y_nodig))
    id_anomaly = id_anomaly.astype(int)
    return x_training,y_training,id_anomaly