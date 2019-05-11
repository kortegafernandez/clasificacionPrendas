#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:50:13 2018

@author: madcas
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils_cifar import *
import cv2


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Crea los placeholders para la sesión.
    
    Argumentos:
    n_H0 -- Escalar, height de la imagen de entrada
    n_W0 -- Escalar, width de la imagen de entrada
    n_C0 -- Escalar, Número de canales de entrada
    n_y -- Escalar, Número de clases
        
    Returna:
    X -- placeholder para los datos de entrada, de tamaño [None, n_H0, n_W0, n_C0] y dtype "float"
    Y -- placeholder para las etiquetas de entrada, de tamaño [None, n_y] y dtype "float"
    """

    #### Haga su código áca ### (≈2 lines)

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    
    ### Fin ###
    
    return X, Y
    
    
def initialize_parameters():
    """
    Inicializa los parámetros (Pesos) para construir la red neuronal convolucional con tensorflow. El tamaño es
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returna:
    parameters -- Un diccionario de tensores que contiene W1, W2
    """
    
    tf.set_random_seed(1)                              # 
        
    #### Haga su código áca ### (≈2 lines)
        
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
#    W3 = tf.get_variable("W3", [2, 2, 16, 24], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
   
    ### Fin ###

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters


def forward_propagation(X, parameters):
    """
    Implementa la propagación hacia adelante del modelo

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Argumentos:
    X -- placeholder de entrada (ejemplos de entrenamiento), de tamaño (input size, number of examples)
    parameters -- Diccionario que contiene los parámetros "W1", "W2" desde initialize_parameters

    Returna:
    Z3 -- Salida de la última unidad LINEAR 
    """
    
    # Obtención de los pesos desde "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    #### Haga su código áca ### 
    
    # CONV2D: stride of 1, padding 'SAME'
    
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    
    # RELU
    
    A1 = tf.nn.relu(Z1)
    
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    # CONV2D: filters W2, stride 1, padding 'SAME'
    
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    
    # RELU
    
    A2 = tf.nn.relu(Z2)
    
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
    # FLATTEN
    
    F = tf.contrib.layers.flatten(P2)
    
#    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
#    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    
    Z3 = tf.contrib.layers.fully_connected(F, 10, None)
    
    ### Fin ###

    return Z3
    

def coompute_cost(Z3, Y):
    """
    Calcula la función de costo
    
    Argumentos:
    Z3 -- Salida del forward propagation (Salida de la última unidad LINEAR), de tamaño (6, Número de ejemplos)
    Y -- placeholders con el vector de etiquetas "true", del mismo tamaño que Z3

    Returns:
    cost - Tensor de la función de costo
    """
    
    #### Haga su código áca ### (≈2 lines)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    
    ### Fin ###
    
    return cost  
   
 
def model_predict(data_test, y_label, learning_rate = 0.009, num_epochs = 100, minibatch_size = 10, print_cost = True):
    """
    Implementa una Red Neuronal Convolucional de 3-Capas en Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Argumentos:
    learning_rate -- factor de aprendizaje en la optimización
    num_epochs -- Número de epocas en el ciclo de optimización
    minibatch_size -- Tamaño del minibatch
    print_cost -- True: imprime el costo cada 100 epocas
    
    Returna:
    train_accuracy -- Número Real, Accuracy del conjunto de entrenamiento (X_train)
    test_accuracy -- Número Real, Accuracy del conjunto de Test(X_test)
    parameters -- parameters aprendidos por el modelo. Estos pueden ser usados para predecir.
    """
    
    ops.reset_default_graph()                         # Permite correr nuevamente el modelo sin sobreescribir las tf variables
    tf.set_random_seed(1)                             #  (tensorflow seed)
    seed = 3         
    (m, n_H0, n_W0, n_C0) = data_test.shape             
    n_y = y_label.shape[1]                                   # 
    
    # Crear los PlaceHolders
          
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    
    # Inicializar Parámetros
    
    parameters = initialize_parameters()
    
    # Forward propagation: Construir el forward propagation en el grafo de tensorflow
    
    Z3 = forward_propagation(X, parameters)
            
    saver = tf.train.Saver()
     
    # Iniciar la sesión 
    with tf.Session() as sess:
        
        saver.restore(sess, "G:\\Mi unidad\\ITM\\Extensión\\DeepLearning\\2018_2\\RedesConvolucionales_Cambios\\model_softmax_Cifar3Capas_batch.ckpt")        
        print("Model restored.")        
               
        predict_op = tf.nn.softmax(Z3)  # Apply softmax to logits
               
        # Calcular la predicción 
        top_n_predictions=3
        prediccion = sess.run(tf.nn.top_k(tf.nn.softmax(Z3), top_n_predictions),feed_dict={X: data_test, Y: y_label})
        print("prediccion:", prediccion)
        
        display_image_predictions(data_test, y_label, prediccion)
        
                
        return prediccion
    
    
# Test
        
X_test, Y_test=load_test_cifar()
index=200
X_test=X_test[index:index+4,...]
#X_test=X_test[np.newaxis,...]

Y_test=Y_test[:,index:index+4]
#Y_test=Y_test[..., np.newaxis]

prediccion=model_predict(X_test, Y_test)

