# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:25:44 2018

@author: Familiamadcas2
"""
import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image

parent_dir='C:\\Users\\Familiamadcas2\\Documents\\Base De Datos Animales'
file_ext='*.jpg'
classes_id={
              'aguilas':0,
              'ardillas':1,
              'buho':2,
              'caballos':3,
              'cocodrilos':4,
              'gatos':5,
              'guacamaya':6,
              'jirafas':7,
              'mariposa':8,
              'mono':9,
              'oso':10,
              'oveja':11,
              'pantera (1)':12,
              'peces':13,
              'perro':14,
              'renos':15,
              'rinoceronte':16,
              'sariguella':17,
              'serpiente':18,
              'tigres':19,
              'toros':20
          }
def label2num(x):
  if x in classes_id:
    return classes_id[x]
  else:
    return(21)
    

def extract_features(parent_dir,file_ext):
  imgs = []
  labels = []
  for sub_dir in os.listdir(parent_dir):
      print(sub_dir)
      for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        print(fn)
        try:
          img = Image.open(fn)
          if (img.size[0] >= 224):
            etiqueta=label2num(sub_dir)
            img=img.resize((224,224),Image.ANTIALIAS)
            im=np.array(img)
            if im.shape[2]==3:
              imgs.append(im)
              labels.append(etiqueta)
            else:            
              print(fn,im.shape) 
        except:
          print('Error',fn)
  features = np.asarray(imgs).reshape(len(imgs),224,224,3)
  return features, np.array(labels,dtype = np.int)

features,labels = extract_features(parent_dir, file_ext)

# Si se requiere se hace alguno de los siguientes procesos.
# Mesclar aleatoriamente la base de datos
x1, y1 = shuffle(features, labels)

# O Separarla en entrenamieto y prueba (o si es necesario en validación)
samples=y1.size
y1=y1.reshape((samples,1))

offset = int(x1.shape[0] * 0.80)
X_train, Y_train = x1[:offset], y1[:offset]
X_test, Y_test = x1[offset:], y1[offset:]
Y_test = np.array(Y_test)
Y_train = np.array(Y_train)
# Se puede adicionar el proceso que requieran para su base de datos

# Este sería el proceso para guardar en un formato h5 los datos
with h5py.File('animales_train_dataset.h5','w') as h5data:
    h5data.create_dataset('train_set_x',data=X_train)
    h5data.create_dataset('train_set_y',data=X_test)
with h5py.File('animales_test_dataset.h5','w') as h5data:
    h5data.create_dataset('test_set_x',data=Y_train)
    h5data.create_dataset('test_set_y',data=Y_test)