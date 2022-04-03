# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:47:50 2020

@author: liorr
"""
import pickle
import re,nltk,os
import tensorflow as tf
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer


def prep(text):
    def read():
        nonlocal data
        with  open("revs_prep.pkl","rb") as f:
            data = pickle.load(f)
            data = data["dict"].word2idx
            
            
    def decontractiate():
        nonlocal text         
        text = re.sub("[^a-z ]+", '', text.lower().replace('.',' ').replace(',',' '))
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub("cannot", " can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text= re.sub(r"sunday", "", text)
        text= re.sub(r"monday", "", text)
        text= re.sub(r"tuesday", "", text)
        text= re.sub(r"wednesday", "", text)
        text= re.sub(r"thursday", "", text)
        text= re.sub(r"friday", "", text)
        text= re.sub(r"saturday", "", text)
    
    def lemmatize():
        nonlocal text
        text=list(map(lambda x: wordnet_lemmatizer.lemmatize(x), text.split()))
    def tranlsate():
        nonlocal text
        text=[data[w] for w in text if w not in stop_words and w  in data.keys()]
    
    def prep():
        if not data:
            read()
            
        decontractiate()
        lemmatize()
        tranlsate()
        return np.array(text+[0 for i in range(100-len(text))])
        
        

    nltk.download('wordnet')#if the wordnet is not uptodate or doesnot exsists in the computer then downloading it from nltk servers we will nedd this for lemma
    nltk.download('stopwords')#if the stopwords vocab is not uptodate or doesnot exsists in the computer downloading then it from nltk servers 
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {"no","not", "nor"}         
    data = None
    return{"prep":prep}






def recommend():
    def load():
        nonlocal data,business_ids
        with open("btable","rb") as f:
            data = pickle.load(f)
        business_ids = np.array(list(set(data.business)))
    def rmse(y_true, y_pred):
         """
         rmse evaluation metric
         """
         return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))
        
    def generate_recommentation(user_id):
        """
        for old user we will load the model, pass throw the model all the business and
        then show him the top 3 recommindations
        """
        model = tf.keras.models.load_model('finalReg.h5',custom_objects={'rmse':rmse})
        #we need to pass into the model the user id with all the  anime ids, so it can make perdictions to the user
        # in order to fo this we need to build user id's list in the len of the anime_ids
        user_ids=np.array([user_id  for _ in range(len(business_ids))])
        
        predictions = np.array([x[0] for x in model.predict([user_ids,business_ids])])
        ids = (-predictions).argsort()[:5]
        print(set(data[data.business.isin(ids)]["name"].values))

    def predict(users):
        """
        for each user the function  checks if the user is new or not and activates the corresponding function
        """
        for user in users:
            if user not in set(data["user"].tolist()):
                raise Exception("user is dead")
            else:
                generate_recommentation(user)
    def sentiment(text):
        model=tf.keras.models.load_model('finalSentiment.h5',custom_objects={'leaky_relu':tf.nn.leaky_relu})
        prepare=prep(text)
        text = prepare['prep']()
        prec = model.predict(np.array([text]))[0][0]
        print(prec)
        

    data,business_ids=None,None
    if not data: load()
    return {"recommend":predict,"sentiment":sentiment}
    
rec=recommend()

