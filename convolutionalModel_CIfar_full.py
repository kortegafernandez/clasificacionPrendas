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
from cnn_utils_cifar import *

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
    
    
#X, Y = create_placeholders(64, 64, 3, 6)
#print ("X = " + str(X))
#print ("Y = " + str(Y))

#######  Esto debería dar el Resultado ################

#X = Tensor("Placeholder_2:0", shape=(?, 64, 64, 3), dtype=float32)
#Y = Tensor("Placeholder_3:0", shape=(?, 6), dtype=float32)

#######################################################
    
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
    
#tf.reset_default_graph()
#
#with tf.Session() as sess_test:
#    parameters = initialize_parameters()
#    init = tf.global_variables_initializer()
#    sess_test.run(init)
#    print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
    
    
#######  Esto debería dar el Resultado ################

#W1 = [ 0.00131723  0.1417614  -0.04434952  0.09197326  0.14984085 -0.03514394
# -0.06847463  0.05245192]
#W2 = [-0.08566415  0.17750949  0.11974221  0.16773748 -0.0830943  -0.08058
# -0.00577033 -0.14643836  0.24162132 -0.05857408 -0.19055021  0.1345228
# -0.22779644 -0.1601823  -0.16117483 -0.10286498]

#######################################################


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
    
#tf.reset_default_graph()
#
#with tf.Session() as sess:
#    np.random.seed(1)
#    X, Y = create_placeholders(64, 64, 3, 6)
#    parameters = initialize_parameters()
#    Z3 = forward_propagation(X, parameters)
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#    print("Z3 = " + str(a))
#    print("Z3 = " + str(Z3.shape))

#######  Esto debería dar el Resultado ################

#Z3 = [[ 1.4416984  -0.24909666  5.450499   -0.2618962  -0.20669907  1.3654671 ]
# [ 1.4070846  -0.02573211  5.08928    -0.48669922 -0.40940708  1.2624859 ]]
#Z3 = (?, 6)

#######################################################


def compute_cost(Z3, Y):
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
    
#tf.reset_default_graph()
#
#with tf.Session() as sess:
#    np.random.seed(1)
#    X, Y = create_placeholders(64, 64, 3, 6)
#    parameters = initialize_parameters()
#    Z3 = forward_propagation(X, parameters)
#    cost = compute_cost(Z3, Y)
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
#    print("cost = " + str(a))
    
    
#######  Esto debería dar el Resultado ################

#cost = 4.6648693

#######################################################
    
 
def model_cifar(learning_rate = 0.009, num_epochs = 5, minibatch_size = 64, print_cost = True):
    """
    Implementa una Red Neuronal Convolucional de 3-Capas en Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Argumentos:
    X_train -- Conjunto de entrenamiento, de tamaño (None, 64, 64, 3)
    Y_train -- Etiquetas del conjunto de entrenamiento, de tamaño (None, n_y = 6)
    X_test -- Conjunto de datos de Test, de tamaño (None, 64, 64, 3)
    Y_test -- Etiquetas del conjunto de Test, de tamaño  (None, n_y = 6)
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
    seed = 3                                          # 
#    (m, n_H0, n_W0, n_C0) = X_train.shape   

    (n_H0, n_W0, n_C0) = (32, 32, 3)          
#    n_y = Y_train.shape[1]  

    n_y = 10                           
    costs = []                                        # Para almacenar el costo
    
    # Crear los PlaceHolders
          
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    
    # Inicializar Parámetros
    
    parameters = initialize_parameters()
    
    # Forward propagation: Construir el forward propagation en el grafo de tensorflow
    
    Z3 = forward_propagation(X, parameters)
        
    # Cost function: Incluir la  función de costo en el grafo de tensorflow
        
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define el optimizador. Usar AdamOptimizer para minimizar el costo.

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # Inicializar todas las variables globales
    
    init = tf.global_variables_initializer()
    
    # genera el objeto para guardar el modelo entrenado
    
    saver = tf.train.Saver()
     
    # Iniciar la sesión 
    with tf.Session() as sess:
        
        # Run init
        sess.run(init)
        
        # Loop de entrenamiento
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = 0 # número de minibatches de tamaño minibatch_size en el conjunto de entrenamiento          seed = seed + 1
#            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            n_batches = 5
            
            for minibatch_i in range(1, n_batches + 1):
                
                for minibatch_X, minibatch_Y in load_preprocess_training_batch(minibatch_i, minibatch_size):
                    
                    minibatch_X= minibatch_X.reshape(len(minibatch_X),32,32,3)
                    minibatch_Y=np.array(minibatch_Y)
                    minibatch_Y=minibatch_Y[:, np.newaxis]
                    minibatch_Y = convert_to_one_hot(minibatch_Y, 10).T
 
                    # Seleccionar un minibatch
                    
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    
                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
    
                    _ , temp_cost = sess.run([optimizer, cost], {X: minibatch_X, Y: minibatch_Y})
                    
                    num_minibatches+=1                  
                    minibatch_cost += temp_cost
                    
                minibatch_cost/= num_minibatches
                num_minibatches=0
                
            # Imprime el costo
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # Graficar la función de costo
        plt.figure(2)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calcular las predicciones correctas
        predict_op = tf.nn.softmax(Z3)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(predict_op, 1), tf.argmax(Y, 1)) 
        
#        # Calcular la predicción sobre el conjunto de test 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy=0
        num_minibatches_train=0
        
        for minibatch_i in range(1, n_batches + 1):
            for minibatch_X, minibatch_Y in load_preprocess_training_batch(minibatch_i, minibatch_size):
                        
                minibatch_X= minibatch_X.reshape(len(minibatch_X),32,32,3)
                minibatch_Y=np.array(minibatch_Y)
                minibatch_Y=minibatch_Y[:, np.newaxis]
                minibatch_Y = convert_to_one_hot(minibatch_Y, 10).T
                num_minibatches_train+=1 
                train_accuracy += sess.run(accuracy, {X:minibatch_X, Y:minibatch_Y})
        
        train_accuracy=  train_accuracy/num_minibatches_train
        
        num_minibatches_test=0
        test_accuracy = 0

        for batch_valid_features, batch_valid_labels in load_preprocess_test_batch(minibatch_size):
            batch_valid_features= batch_valid_features.reshape(len(batch_valid_features),32,32,3)
            batch_valid_labels=np.array(batch_valid_labels)
            batch_valid_labels=batch_valid_labels[:, np.newaxis]
            batch_valid_labels = convert_to_one_hot(batch_valid_labels, 10).T
            test_accuracy += sess.run(accuracy, {X:batch_valid_features, Y:batch_valid_labels})
            num_minibatches_test=num_minibatches_test+1
            
        test_accuracy=  test_accuracy/num_minibatches_test
        
        
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        save_path = saver.save(sess, "G:\\Mi unidad\\ITM\\Extensión\\DeepLearning\\2018_2\\RedesConvolucionales_Cambios\\model_softmax_Cifar3Capas_batch.ckpt")
        print("Model saved in path: %s" % save_path)
                
#                
        return train_accuracy, test_accuracy, parameters
    
    
_, _, parameters = model_cifar()




