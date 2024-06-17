import math

from tensorflow.keras import layers,losses
import numpy as np
import os
from tensorflow.keras.models import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import AE_architectures
import matplotlib.pyplot as plt
import query_collection

from sklearn.metrics import roc_auc_score,roc_curve, auc,average_precision_score
from scipy.spatial.distance import euclidean

from pylatex import Document, Section, Subsection, Command, Figure, MiniPage
from pylatex.utils import italic, NoEscape

import tensorflow as tf
import random

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_oversampling(ids_asked_queries, total_size):
    n_queries = ids_asked_queries.shape[0];
    ids_sampling = ids_asked_queries[n_queries-1]*np.ones(total_size,dtype=int);
    #DISTRIBUZIONE UNIFORME
    samples_per_query = total_size//n_queries;
    start = 0;
    for query in ids_asked_queries:
        ids_sampling[start:start + samples_per_query] = query;
        start = start + samples_per_query;
    return ids_sampling;

def launch(x_training, oracle, labels, budget, n_query, classe, conf ,alpha, dir, anomalie, AE_type=None, n_candidates=10 ,latent_dim=32, F = '_', EP=(25,10), batch_size=32, intermediate_dim=128, verb=0):
    dim = x_training[0].shape
    flat_dim = x_training[0].flatten().shape[0]

    if AE_type=='shallow':
        autoencoder = AE_architectures.Shallow_Autoencoder(dim,flat_dim,latent_dim)
    if AE_type=='deep':
        autoencoder = AE_architectures.Deep_Autoencoder(dim,flat_dim,intermediate_dim,latent_dim)
    if AE_type=='conv':
        autoencoder = AE_architectures.Conv_Autoencoder(dim)
    if AE_type=='bigconv':
        autoencoder = AE_architectures.Big_Conv_Autoencoder(dim)
    if AE_type=='pca':
        autoencoder = AE_architectures.PCA_Autoencoder(dim,flat_dim,latent_dim)

    y_training = np.int32(np.zeros_like(oracle))

    b = 0

    result_vector = np.full(budget, 10)
    result_vec_indi = np.full(budget, 0)

    weights_inits = False

    selected = np.zeros(x_training.shape[0])==1

    if not conf['purity']:
        alpha = 1

    if conf['weighting']:
        Lambda_a = None
        Lambda_n = None
    else:
        Lambda_a = 1
        Lambda_n = 1

    numero_copie = math.ceil((x_training.shape[0] / batch_size) * 0.3)
    x_training_augumented = np.zeros([x_training.shape[0]+numero_copie]+list(x_training.shape[1:len(x_training.shape)]))
    y_training_augumented = np.int32(np.zeros(y_training.shape[0]+numero_copie))

    x_training_augumented[:x_training.shape[0]] = x_training
    y_training_augumented[:y_training.shape[0]] = y_training


    '''indici_casuali = np.random.choice(x_training.shape[0], size=numero_copie, replace=False)

    # Inserisci le copie in posizioni casuali
    for indice in indici_casuali:
        img_expanded = np.expand_dims(x_training[0], axis=0)
        x_training_augumented = np.insert(x_training_augumented, indice, img_expanded, axis=0)
        y_training_augumented = np.insert(y_training_augumented, indice, -1)'''

    #r = np.ones(x_training_augumented.shape[0])

    while b < budget and n_query != 0:

        modello_salvato = f"{dir}/modello_{n_query}_{EP[0]}_{EP[1]}_{b}.h5"

        if b == 0:
            # PRIMO ADDESTRAMENTO
            decoded_img, rec_err, f_rec_err = ae_sad(x_training, None, y_training, x_training,
                                                        y_training, autoencoder, weights_inits,Lambda_a=Lambda_a,Lambda_n=Lambda_n, EP=EP[0],
                                                        verb=verb)
            weights_inits = True

        else:
            # ADDESTRAMENTO ARCHITETTURA
            if conf['oversampling']:
                decoded_img, rec_err, f_rec_err = ae_sad(x_training, x_train_alpha, y_training, x_training_augumented,
                                                                y_training_augumented, autoencoder, weights_inits,Lambda_a=Lambda_a,Lambda_n=Lambda_n, EP=EP[1],
                                                                verb=verb)
            else:
                decoded_img, rec_err, f_rec_err = ae_sad(x_training, x_train_alpha, y_training, x_training,
                                                            y_training, autoencoder, weights_inits,Lambda_a=Lambda_a,Lambda_n=Lambda_n, EP=EP[1],
                                                            verb=verb)


        # CANDIDATE SELECTION
        if conf['indecision']:
            result_file = os.path.join(dir, f'rec_err_{b}.npy')
            np.save(result_file, rec_err)

            z = np.where(rec_err < f_rec_err, rec_err, f_rec_err)

            result_file1 = os.path.join(dir, f'min_rec_frec_{b}.npy')
            np.save(result_file1, z)

        else:
            result_file = os.path.join(dir, f'rec_err_{b}.npy')
            np.save(result_file, rec_err)

            z = rec_err

        j=0
        k=0

        if b >= budget-n_query-1:
            n_query = budget-b-1

        while k < n_query and j < y_training.shape[0]:

            # QUERY COLLECTION
            indix = query_collection.choice(selected_indices=selected, x_training=x_training, y_training= y_training,decoded_imgs=decoded_img, z_all=z, n_candidates=n_candidates, config = conf, labels=labels)

            # QUERY THE USER
            y_training[indix] = oracle[indix]
            result_vec_indi[b + k] = indix
            print(f"The class of the query is {labels[indix]} - The index is {indix} - The normal class is {classe}")
            if oracle[indix] == -1:
                result_vector[b + k] = labels[indix]

            # TRAINING SET BUILDING
            ids_sorted_unlabeled = np.argsort(rec_err)
            ids_sorted_unlabeled = ids_sorted_unlabeled[y_training[ids_sorted_unlabeled] == 0]

            x_train_alpha = ids_sorted_unlabeled[:math.floor(ids_sorted_unlabeled.shape[0] * alpha)]

            ids_asked_queries = np.where(y_training != 0)[0];
            ids_sampling = get_oversampling(ids_asked_queries, numero_copie);

            x_training_augumented[x_training.shape[0]:x_training_augumented.shape[0]] = x_training[ids_sampling];
            y_training_augumented[y_training.shape[0]:y_training_augumented.shape[0]] = y_training[ids_sampling];

            indice_stamp = indix
            selected[indix] = True

            k += 1
            j += 1

        print(f"budget corrente: {b+1}/{budget}")
        if (b < 10) or ((b+1) % 10 == 0):
            autoencoder.save_weights(modello_salvato)
            print("Modello salvato !!!")

        b = b + n_query

    print(f"---{b+1} query presented to the expert---")

    _, rec_err, f_rec_err = ae_sad(x_training, x_train_alpha, y_training, x_training_augumented,
                                                    y_training_augumented, autoencoder, weights_inits,Lambda_a=Lambda_a,Lambda_n=Lambda_n, EP=EP[1],
                                                    verb=verb)
    return rec_err, f_rec_err, result_vector, result_vec_indi

def ae_sad(x_training, ids_x_alpha, y_training, x_training_augumented, y_training_augumented,  autoencoder=None, weights_inits = False,latent_dim=32,Lambda_a=None, Lambda_n=None,F = '_',EP=1000,batch_size=32,intermediate_dim=128, verb=0):

    dim = x_training[0].shape
    flat_dim = x_training_augumented[0].flatten().shape[0]

    if weights_inits == False:
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        
    if ids_x_alpha is None:
        ids_x_alpha = np.arange(x_training.shape[0])

    n_x_alpha = len(ids_x_alpha);
    if Lambda_a is None:
        if np.sum(y_training == -1) != 0:
            Lambda_a = n_x_alpha/np.sum(y_training == -1)
        else:
            Lambda_a = 1

    if Lambda_n is None:
        if np.sum(y_training == 1) != 0:
            Lambda_n = n_x_alpha/np.sum(y_training == 1)
        else:
            Lambda_n = 1


    ids_oversample = np.arange(x_training.shape[0], x_training_augumented.shape[0])
    ids_queries = np.where(y_training != 0)[0]
    ids_training = np.concatenate((ids_x_alpha, ids_oversample, ids_queries))
    

    x_train = x_training_augumented[ids_training];
    y_train = y_training_augumented[ids_training];

    ids_normal = y_train == 1;
    ids_anomal = y_train == -1;

#################################à
#################################à
#################################à
#################################à
    weights = 10*np.ones((y_train.shape[0],1));
#################################à
#################################à
#################################à
#################################à
    weights[ids_normal,0] = 0.1*Lambda_n;
    weights[ids_anomal,0] = Lambda_a;
    x_target = x_train.copy();

    if F == '_':
        x_target[ids_anomal] = 1 - x_target[ids_anomal]
    if F == '01':
        x_target[ids_anomal] = np.where(x_target[ids_anomal]>0.5,0,1)
    if F == '_05':
        x_target[ids_anomal] = np.where(x_target[ids_anomal]>0.5,
                                                       x_target[ids_anomal]+0.5,
                                                       x_target[ids_anomal]-0.5)

    autoencoder.fit(x_train, x_target,
                    epochs=EP,
                    shuffle=True, batch_size=batch_size, sample_weight=weights, verbose=verb)

    ### -TEST- ###
    encoded_imgs = autoencoder.encoder(x_training).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    rec_err = np.linalg.norm(x_training.reshape(x_training.shape[0], flat_dim) - decoded_imgs.reshape(x_training.shape[0], flat_dim), axis=(1)) ** 2

    if F == '_':
        fx_training = 1 - x_training
    if F == '01':
        fx_training = np.where(x_training>0.5,0,1)
    if F == '_05':
        fx_training = np.where(x_training>0.5, x_training+0.5, x_training-0.5)

    f_rec_err = np.linalg.norm(
        fx_training.reshape(x_training.shape[0], flat_dim) - decoded_imgs.reshape(x_training.shape[0], flat_dim),
        axis=(1)) ** 2

    return decoded_imgs, rec_err, f_rec_err
