# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:37:35 2020

@author: liorr
"""

import re,nltk,os
import dictionary,pickle
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer

class DataSet:
    """
    a class to preprocess the text with the following methods:
        1. stop words removal
        2. contraction expanding
        3. dictionary building
        4. lemmatize
        5. WeightMatrix building
        6. one hot encoding
        
    """
    
    def __init__(self,dictonary = dictionary.Dictionary(),path="3stars rev"):
        self.dictionary = dictonary# building a new dictionary
        with open(path,"rb") as f:
            self.data =pickle.load(f) 
        nltk.download('wordnet')#if the wordnet is not uptodate or doesnot exsists in the computer then downloading it from nltk servers we will nedd this for lemma
        nltk.download('stopwords')#if the stopwords vocab is not uptodate or doesnot exsists in the computer downloading then it from nltk servers 
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {"no","not", "nor"}
        self.managePreparation
        self.buildWeightMatrix
        self.save
        
    
       
    @property   
    def managePreparation(self):
       
         def decontractiate(text):
            """
            an method to decontractiate the text
            """
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
            return text
            

         def split():
             from sklearn.model_selection import train_test_split
             self.data['X_train'], self.data['X_test'], self.data['Y_train'], self.data['Y_test'] = \
                 train_test_split(self.data['X'], self.data['Y'], test_size=0.33, random_state=42)
             del self.data['X'], self.data['Y']
            
         self.data['text']=self.data['text'].apply(decontractiate)
         print("starting lemma")
         self.data['text']=self.data['text'].apply(lambda text: list(map(lambda x: self.wordnet_lemmatizer.lemmatize(x), text.split())))
         print("starting stop words removal")
         self.data['text']=self.data['text'].apply(lambda text: [w for w in text if w not in self.stop_words])

         print("starting dict")
         for revlist in self.data["text"].tolist():
             self.dictionary.add_word(revlist)
         print("starting encoding")
         self.data['text']=self.data['text'].apply(lambda text: [self.dictionary[word] for word in text])
         
         print("starting padding")
         self.data['text']=self.data['text'].apply(lambda text:text+[0]*(100-len(text)) if len(text)<100 else text[:100])   

       #  self.data["stars"]=self.data["stars"].apply(float)
         print("starting spliting and saving")
         self.data={'X':np.array(self.data['text'].tolist()),'Y':np.array(self.data["stars"].tolist()),'max_features':len(self.dictionary)}
         split()


    @property   
    def buildWeightMatrix(self):
        def initFastText():
            nonlocal fastText
            with open(os.path.join(os.getcwd(),"wiki.vec"), 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
                    _, self.emmbedingdim = map(int, f.readline().split())
                    fastText = {}
                    for line in f:
                        tokens = line.rstrip().split(' ')
                        fastText[tokens[0]] = np.asarray(tokens[1:], "float32")
                        
            
        def buildMatrix():
             maching_words = 0
             dataset_size = len(self.dictionary)+1
             self.weights_matrix = np.zeros(shape=(dataset_size,self.emmbedingdim))
             for i,word in enumerate(self.dictionary.word2idx.keys(),1):
                     try:
                        save = fastText[word]
                        maching_words += 1
                     except KeyError:
                        save = np.random.normal(scale=0.6, size=(self.emmbedingdim,))
                     self.weights_matrix[i] = save
                     
                 
             print("pre-treind words: {} randomaly initilaized: {}".format(maching_words,dataset_size))
             
        def saveMatrix():
            with open("weights_matrix {} dim.pkl".format(self.emmbedingdim),"wb") as f:
                pickle.dump(self.weights_matrix,f , 2)
        
        fastText=[]
        initFastText()
        buildMatrix()
        self.data['matrix'] = self.weights_matrix
        self.data['dict']  = self.dictionary
        
    @property
    def save(self):
        with open("revs_prep.pkl","wb") as f:
            pickle.dump(self.data,f)