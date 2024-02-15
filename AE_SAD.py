from tensorflow.keras import layers,losses
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import AE_architectures
import matplotlib.pyplot as plt

def launch(x_training,y_training,x_test,AE_type=None,latent_dim=32,Lambda=None,F = '_',EP=1000,batch_size=32,intermediate_dim=128):
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

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    if Lambda is None:
        Lambda = x_training.shape[0]/np.sum(y_training)
    weights = np.ones(y_training.shape[0])
    weights[np.where(y_training == 1)] = Lambda
    x_target = copy.copy(x_training)
    if F == '_':
        x_target[np.where(y_training == 1)] = 1 - x_target[np.where(y_training == 1)]
    if F == '01':
        x_target[np.where(y_training == 1)] = np.where(x_target[np.where(y_training == 1)]>0.5,0,1)
    if F == '_05':
        x_target[np.where(y_training == 1)] = np.where(x_target[np.where(y_training == 1)]>0.5,
                                                       x_target[np.where(y_training == 1)]+0.5,
                                                       x_target[np.where(y_training == 1)]-0.5)

    weights = weights.reshape((weights.shape[0],1))

    autoencoder.fit(x_training, x_target,
                        epochs=EP,
                        shuffle=True, batch_size=batch_size, sample_weight=weights, verbose=1)


    ### -TEST- ###
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    ### -REC TRAIN- ###
    '''encoded_imgs_tr = autoencoder.encoder(x_training).numpy()
    decoded_imgs_tr = autoencoder.decoder(encoded_imgs_tr).numpy()'''

    rec_err = np.linalg.norm(x_test.reshape(x_test.shape[0], flat_dim) - decoded_imgs.reshape(x_test.shape[0], flat_dim), axis=(1)) ** 2

    return decoded_imgs, rec_err

