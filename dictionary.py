# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:32:39 2019

@author: lior
"""
import pickle

class Dictionary:
    """
    a class that represents the dictionary of the text
    the class gives a unique index to every word
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1
    
    def add_word(self,input):
        """
        Input: word or words 
        InputType: list,tuple or str
        a method to add a word(s),
        the method checks if the word(s) is allredy in the dictinoary
        if not, add it.
        othewith just ignores it
        """
        def add():
            if word not in self.word2idx.keys():
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx+=1
       
        def addSeq():
            nonlocal word
            for word in input:
                add()
        inputType = type(input)
        word = []
        if inputType != list and inputType != str and inputType != tuple:
            raise TypeError ("the type of the input is not allowed")
        if inputType == "str":
            input = input.split()
        addSeq()
            
            
    def __getitem__(self,word):
        try:
            return self.word2idx[word]
        except KeyError:
            print("The word does not exists in the dict")
    
    def __setitem__(self,word,id):
        """
        add (word,id) to the dict iff the (word,id) does not appear in the dict
        """
        if word not in self.word2idx.keys():
            self.word2idx[word] = id 
            self.word2idx[id] = word 
            self.idx+=1

            
    def __len__(self):
        """
        returns the size of the dictionary
        """
        return len(self.word2idx)
    
    def save(self,name="Dict.pkl"):
        """
        a method to save the dictionary to the Hard Drive for future use
        """
        with open(name,"wb") as f:
            pickle.dump(self.__dict__,f , 2)
            
      
    def load(self,filename):
        """
        a method to load the dictionary from the Hard Drive
        """
        self.__dict__.clear() 
        with open(filename,"rb") as f:
            self.__dict__.update(pickle.load(f))
            