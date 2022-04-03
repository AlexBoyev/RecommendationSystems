# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:24:02 2020

@author: liorr
"""
import pickle,tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def preapre_data(set_name="regressionset"):
    def read():
        nonlocal data
        with open(set_name,"rb") as f:
            data = pickle.load(f)
    
    def encode():
        nonlocal users,items,num_users,num_business
        user_enc = preprocessing.LabelEncoder()
        data['user'] = user_enc.fit_transform(data['user_id'].values)
        num_users = data['user'].nunique()
        item_enc = preprocessing.LabelEncoder()
        data['business'] = item_enc.fit_transform(data['business_id'].values)
        num_business = data['business'].nunique()
    
    def split_data():
        nonlocal X_train,X_test,Y_train,Y_test,X,y
        X = data[['user', 'business']].values
        y = data['stars'].values
        X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        X_train, X_test =  [X_train[:, 0], X_train[:, 1]] , [X_test[:, 0], X_test[:, 1]] 

             
    
    def get_data():
        return  X_train,X_test,Y_train,Y_test,num_users,num_business
    def get_full_data():
        return X,y,data
    def prepare():
        read()
        encode()
        split_data()
        save()
    def save():
        with open("full_data","wb") as f:      
            pickle.dump(data,f)
    data,users,items,num_users,num_business, X_train,X_test,Y_train,Y_test,X,y = [], [],[],0,0,0,0,0,0,0,0
    
    return {'read':read,'prepare':prepare,'get_data':get_data}
def rmse(y_true, y_pred):
	return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

def Recommender(**kwargs):
        def bias():
            nonlocal out
            print("in bais")
            u_bis = tf.keras.layers.Embedding(users_amount,1)(u)
            b_bis = tf.keras.layers.Embedding(businesses_amount,1)(b)
            out = tf.keras.layers.Add()([out,u_bis,b_bis])
            out = tf.keras.layers.Dropout(kwargs.get("dropout",0.5))(out)
            out = tf.keras.layers.Flatten()(out)
        def deep_reg():
            nonlocal out
            bias()
            out = tf.keras.layers.Dense(kwargs.get("dense",10),activation=kwargs.get("activation","relu"))(out)
            out = tf.keras.layers.Dense(1)(out)            

            
    
        type = kwargs.get("type")
        users_amount = kwargs.get("users_amount")+1
        businesses_amount = kwargs.get("businesses_amount")+1
        emd = kwargs.get("emd",10)
        embeddings_initializer=kwargs.get("embeddings_initializer",'he_normal')
        embeddings_regularizer=kwargs.get("embeddings_regularizer",tf.keras.regularizers.l2(1e-6))
        
        user, business  = tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Input(shape=(1,))#get user and business inputs
        u = tf.keras.layers.Embedding(users_amount,emd, 
                                     embeddings_initializer=embeddings_initializer,
                                     embeddings_regularizer=embeddings_regularizer)(user)
        b = tf.keras.layers.Embedding(businesses_amount,emd, 
                                     embeddings_initializer=embeddings_initializer,
                                     embeddings_regularizer=embeddings_regularizer)(business)
#    
         
        if not type:
            u=tf.keras.layers.Reshape((emd,))(u)
            b=tf.keras.layers.Reshape((emd,))(b)
        out = tf.keras.layers.Dot(axes=1)([u, b])
        if type == "bias":
            bias()
            out = tf.keras.layers.Dense(1)(out)            
        elif type == "deep":
             deep_reg()   
        

        model = tf.keras.models.Model(inputs=[user, business], outputs=out)
        opt = tf.keras.optimizers.Adam(lr=kwargs.get("lr",0.001))
        model.compile(loss= tf.keras.losses.Huber(), metrics=['mse', 'mae', rmse]
        , optimizer=opt)
        print(model.summary())
        return model


#
#    
def trainer(his="f1",checkpiont='F1.h5',emd=20,type="bias",embeddings_initializer='he_normal',embeddings_regularizer=tf.keras.regularizers.l2(1e-6)):    
    #save the best model
    checkpiont=tf.keras.callbacks.ModelCheckpoint(checkpiont, monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='auto', peroid=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.2,
                                  patience=3, min_lr=0.000001)
    
    
    data=preapre_data()
    data["prepare"]()
    X_train,X_test,Y_train,Y_test,num_users,num_business = data["get_data"]()
    model = Recommender(users_amount=num_users,
                        businesses_amount=num_business,
                        emd=emd,
                        type=type,
                        embeddings_initializer=embeddings_initializer,
                        embeddings_regularizer=embeddings_regularizer)
    history = model.fit(x=X_train, y=Y_train, batch_size=64, epochs=3,
                        verbose=1, validation_split = 0.2,callbacks=[checkpiont,reduce_lr])
    predicts=model.evaluate(X_test,Y_test)
    with open(his,"wb") as f:
        pickle.dump(history.history,f,2)
    return predicts,history.history