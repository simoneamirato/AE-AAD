from tensorflow.keras import layers,losses
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import AE_architectures
import matplotlib.pyplot as plt

def launch(x_training, oracle, budget, n_query,AE_type=None, latent_dim=32, F = '_', EP=1000, batch_size=32, intermediate_dim=128, verb=0):
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

    y_training = np.zeros_like(oracle)

    b = 0
    while b < budget:

        #Controllare ciò che sfora
        _, rec_err, f_rec_err = AEtrain(x_training, y_training, autoencoder)
        z = np.where(rec_err > f_rec_err, rec_err, f_rec_err)
        z = np.argsort(z)[::-1]

        j=0
        k=0
        #da rivedere
        while k < n_query:
           if y_training[z[j]] == 0:
               y_training[z[j]]=oracle[z[j]]
               k+=1
           j+=1
        b+=n_query

    _, rec_err, f_rec_err = AEtrain(x_training, y_training, autoencoder)

    return rec_err


def AEtrain(x_training,y_training,autoencoder=None,latent_dim=32,Lambda_a=None, Lambda_n=None,F = '_',EP=1000,batch_size=32,intermediate_dim=128, verb=0):
    dim = x_training[0].shape
    flat_dim = x_training[0].flatten().shape[0]
    '''

    if AE_type=='shallow':
        autoencoder = AE_architectures.Shallow_Autoencoder(dim,flat_dim,latent_dim)
    if AE_type=='deep':
        autoencoder = AE_architectures.Deep_Autoencoder(dim,flat_dim,intermediate_dim,latent_dim)
    if AE_type=='conv':
        autoencoder = AE_architectures.Conv_Autoencoder(dim)
    if AE_type=='bigconv':
        autoencoder = AE_architectures.Big_Conv_Autoencoder(dim)
    if AE_type=='pca':
        autoencoder = AE_architectures.PCA_Autoencoder(dim,flat_dim,latent_dim)'''

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    if Lambda_a is None:
        if np.sum(y_training == -1)!=0:
            Lambda_a = x_training.shape[0]/np.sum(y_training == -1)
        else:
            Lambda_a = 1

    if Lambda_n is None:
        if np.sum(y_training == 1) != 0:
            Lambda_n = x_training.shape[0]/np.sum(y_training == 1)
        else:
            Lambda_n = 1

    weights = np.ones(y_training.shape[0])
    weights[np.where(y_training == 1)] = Lambda_n
    weights[np.where(y_training == -1)] = Lambda_a
    x_target = copy.copy(x_training)
    if F == '_':
        x_target[np.where(y_training == -1)] = 1 - x_target[np.where(y_training == -1)]
    if F == '01':
        x_target[np.where(y_training == -1)] = np.where(x_target[np.where(y_training == -1)]>0.5,0,1)
    if F == '_05':
        x_target[np.where(y_training == -1)] = np.where(x_target[np.where(y_training == -1)]>0.5,
                                                       x_target[np.where(y_training == -1)]+0.5,
                                                       x_target[np.where(y_training == -1)]-0.5)

    weights = weights.reshape((weights.shape[0],1))

    autoencoder.fit(x_training, x_target,
                    epochs=EP,
                    shuffle=True, batch_size=batch_size, sample_weight=weights, verbose=verb)

    ### -TEST- ###
    encoded_imgs = autoencoder.encoder(x_training).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    rec_err = np.linalg.norm(x_training.reshape(x_training.shape[0], flat_dim) - decoded_imgs.reshape(x_training.shape[0], flat_dim), axis=(1)) ** 2

    fx_training = copy.copy(x_training)

    if F == '_':
        fx_training = 1 - fx_training
    if F == '01':
        fx_training = np.where(fx_training>0.5,0,1)
    if F == '_05':
        fx_training = np.where(fx_training>0.5, fx_training+0.5, fx_training-0.5)

    f_rec_err = np.linalg.norm(
        fx_training.reshape(fx_training.shape[0], flat_dim) - decoded_imgs.reshape(x_training.shape[0], flat_dim),
        axis=(1)) ** 2

    return decoded_imgs, rec_err, f_rec_err

