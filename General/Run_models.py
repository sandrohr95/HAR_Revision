# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:20:09 2020

@author: sandrooo
"""
import numpy as np
import General.DL_models as dl_models
from sklearn.model_selection import StratifiedShuffleSplit,RepeatedStratifiedKFold, StratifiedKFold
import tensorflow
from tensorflow.python.keras import backend as K
import General.UTILS as pl
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def run_CNNLSTM(X_train, Y_train, X_test, Y_test, n_steps, n_length, epochs, batch_size):
    
    n_features = X_train.shape[2]
    
    """ create model """
    model_name, model = dl_models.CNNLSTM(X_train, Y_train, n_steps, n_length)
    
    """ Input shape ConvLSTM2D """
    X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
    
    """ entrenamos el modelo """
    model = fit_model_with_cross_validation(model,X_train, Y_train, epochs, batch_size)
    
    """ evaluate the model """
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print("ACCURACY: "+ str(accuracy))
        
    """ SUMMARY OF THE MODEL """
    model.summary()
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    return loss, accuracy, model_name
    
def run_CNN_LSTM(X_train, Y_train, X_test, Y_test, n_steps, n_length, epochs, batch_size):
     
    n_features = X_train.shape[2]
    
    """ create model """
    model_name, model = dl_models.CNN_LSTM(X_train, Y_train, n_steps, n_length)
    
    """ Input shape Timedistributed layer Conv1D """
    X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
    
    """ entrenamos el modelo """
    model = fit_model_with_cross_validation(model,X_train, Y_train, epochs, batch_size)
    
    """ evaluate the model """
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print("ACCURACY: "+ str(accuracy))
        
    """ SUMMARY OF THE MODEL """
    model.summary()
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    return loss, accuracy, model_name
    
def run_LSTM(X_train, Y_train,  X_test, Y_test, epochs, batch_size):
        
    """ create model """
    model_name, model = dl_models.LSTM_model(X_train, Y_train)
    
    """ entrenamos el modelo """
    model = fit_model_with_cross_validation(model,X_train, Y_train, epochs, batch_size)
    
    """ evaluate the model """
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print("ACCURACY: "+ str(accuracy))
        
    """ SUMMARY OF THE MODEL """
    model.summary()
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    return loss, accuracy, model_name

def run_Bidirectional_LSTM(X_train, Y_train,X_test, Y_test, epochs, batch_size):
        
    """ create model """
    model_name, model = dl_models.createBidirectionalLSTM(X_train, Y_train)
    
    """ entrenamos el modelo """
    model = fit_model_with_cross_validation(model,X_train, Y_train, epochs, batch_size)
    
    """ evaluate the model """
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print("ACCURACY: "+ str(accuracy))
        
    """ SUMMARY OF THE MODEL """
    model.summary()
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    return loss, accuracy, model_name

def run_convnet(X_train, Y_train, X_test, Y_test, epochs, batch_size):
        
    """ create model """
    print("create model")
    model_name, model = dl_models.convnet(X_train, Y_train)
    
    """ entrenamos el modelo """
    print("entrenamos el modelo")
    model = fit_model_with_cross_validation(model,X_train, Y_train, epochs, batch_size)
    
    """ evaluate the model """
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print("ACCURACY: "+ str(accuracy))
    print("Test loss: "+ str(loss))
        
    """ SUMMARY OF THE MODEL """
    model.summary()
    
    """ Prediction and Confusion Matrix """
    print("Prediction and Confusion Matrix")
    predict_confusion_matrix(model, X_test, Y_test, Y_train,"CNN")
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    return loss, accuracy, model_name

def run_Bi_LSTM(X_train, Y_train, X_test, Y_test, epochs, batch_size):
        
    """ create model """
    model_name, model = dl_models.Simple_Bi_LSTM(X_train, Y_train)
    
    """ entrenamos el modelo """
    model = fit_model_with_cross_validation(model,X_train, Y_train, epochs, batch_size)
    
    """ evaluate the model """
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print("ACCURACY: "+ str(accuracy))
        
    """ SUMMARY OF THE MODEL """
    model.summary()
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    return loss, accuracy, model_name
    
def fit_model_with_cross_validation(model,X_train, Y_train, epochs, batch_size):
    
    """ Use cross validation only for training with the train set data"""
    
    """ Important to use stratified data for class imbalance 
        mantiene la misma proporción de todas las clases para entrenar el modelo
    """
    
    skf = StratifiedShuffleSplit(n_splits=10, random_state=0,test_size=0.15)
    for train_index, test_index in skf.split(X_train, Y_train):
        xTrain=X_train[train_index]
        yTrain=Y_train[train_index]
        xTest=X_train[test_index]
        yTest=Y_train[test_index]
        model.fit(xTrain, yTrain,validation_data=(xTest, yTest),
                  epochs=epochs, batch_size=batch_size, verbose=1)
        break
        
    
    return model
    
def del_model(model):
    
    # Delete the Keras model with these hyper-parameters from memory.
    del model
        
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.reset_default_graph()


def predict_confusion_matrix(pred, xTest, yTest,name):
    
    class_pred=np.zeros(pred.shape)
    i=0
    for row in pred:
        class_pred[i, np.argsort(-row)[0]]=1.
        i+=1
    i=0
#     for column in class_pred.T:
#         print('Class '+str(i))
#         print(classification_report(yTest[:,i],column))
#         i+=1
    # con argmax devolvemos el único valor que devuelve 1 en ese array, para cada uno de los resultados
    # Ejemplo: [0 0 0 0 0 0 0 1 0 0 0], dará el índice donde se encuentre el valor 1
    # MIRAR CONFUSIÓN MATRIX IN DEEP LEARNING MODELS WITH KERAS
    
    labels = ["Downstairs","Running","Sitting","Standing","Upstairs","Walking"]

    cm= confusion_matrix(yTest.argmax(axis=1), class_pred.argmax(axis=1))
    pl.plot_confusion_matrix(cm, labels ,name)