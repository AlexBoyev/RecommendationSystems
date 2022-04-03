# -*- coding: utf-8 -*-
"""
@author: Lior Reznik
"""
from preProcess import DataSet

import numpy as np
import pickle
import tensorflow as tf

class Trainer:
    """""
    class to train or evalute model
    """""

    def __init__(self,preprocess=DataSet, load=None,path=None):
        """""
        :param load->a pertained model to evaluate
        :param preprocess-> preprocessed data to train on or if load is not none then a data to test on
        """""
        def read():
            with open(path,'rb') as f:
                self.preprocess=pickle.load(f)
        
        if load:
            self.model = tf.keras.models.load_model(load, custom_objects={'leaky_relu': tf.nn.leaky_relu})
            read()
                
        else:
            if path:
                read()
            else:
                self.preprocess=DataSet().data
            self.train
        self.evaluate
        
    

    @property
    def train(self):
      """""
        train the model on preprocessed data
      """""

      self.model = tf.keras.models.Sequential([
      tf.keras.layers.Embedding(self.preprocess["max_features"]+1, 300,weights=[self.preprocess["matrix"]],input_shape=(100,),embeddings_regularizer=tf.keras.regularizers.l2(1e-4)),
      tf.keras.layers.Conv1D(filters=300, kernel_size=3, padding='same', activation='relu'),
      tf.keras.layers.GlobalMaxPooling1D(),
      tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(3,activation="softmax")

    ])
      early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')
       #save the best model
      checkpiont=tf.keras.callbacks.ModelCheckpoint('SENTI11.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                save_weights_only=False, mode='auto', period=1)


      self.model.compile(optimizer= "RMSprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=['acc'], run_eagerly=True)

      self.model.summary()
      self.history=self.model.fit(self.preprocess['X_train'], self.preprocess['Y_train'],batch_size=128, validation_split=0.2, epochs=50, callbacks=[checkpiont,early_stop])
 
                            
    @property
    def evaluate(self,name="model1"):
        """""
        evaluate the model on test data
        """""
        from sklearn.metrics import classification_report
        predicts = [np.argmax(x) for x in self.model.predict(self.preprocess['X_test'])]
        report=classification_report(predicts,self.preprocess['Y_test'])
        with open('history{}'.format(name),"wb") as f:
          pickle.dump(self.history.history,f)
        with open("report{}".format(name),"wb") as f:
          pickle.dump(report,f)

        print(report)

        


 
import os 
path=os.path.join(os.getcwd(),"FINAL1.h5")
t=Trainer(path="revs_prep.pkl")

  
#class OpinionMining(tf.keras.Model):
#
#    def __init__(self):
#        super(OpinionMining, self).__init__()
#        self.block_1 = ResNetBlock()
#        self.block_2 = ResNetBlock()
#        self.global_pool = layers.GlobalAveragePooling2D()
#        self.fully_connected=Dense(128,"linear")
#        self.classifier = Dense(5,"sigmoid")
#
#    def call(self, inputs):
#        x = self.block_1(inputs)
#        x = self.block_2(x)
#        x = self.global_pool(x)
#        return self.classifier(x)