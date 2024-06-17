import math

import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.datasets import mnist
import create_datasets
import matplotlib.pyplot as plt
import AE_architectures
import copy

import os
from PIL import Image

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def euclidean_distance(img1, img2):
    img1 =img1.flatten()
    img2 = img2.flatten()
    return np.linalg.norm(img1 - img2)

def average_distance(images, target):
    distances = [euclidean_distance(target, img) for img in images]
    return np.mean(distances)

def choice(selected_indices, x_training, y_training, decoded_imgs, z_all, n_candidates, config, labels):

    z_ids = np.array([*range(z_all.shape[0])])
    z = z_all[~selected_indices]
    z_ids = z_ids[~selected_indices]
    z_ids = z_ids[np.argsort(z)]
    z_ids = z_ids[::-1]
    z = np.sort(z)[::-1]

    indix = z_ids[0]
    if not config['heterogeneity'] or np.sum(selected_indices)==0:
        return indix

    maxim = average_distance(x_training[selected_indices], x_training[z_ids[0]])
    for w in range(n_candidates):
        dist = average_distance(x_training[selected_indices], x_training[z_ids[w]])
        if dist > maxim:
            maxim = dist
            indix = z_ids[w]

    return indix
