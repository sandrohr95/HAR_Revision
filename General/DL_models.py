from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import Model
# from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import ConvLSTM2D
from keras.layers import LSTM
from keras.layers import ThresholdedReLU
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

def CNNLSTM(trainX, trainy, n_steps, n_length):
    model_name = "CNNLSTM"
    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_name, model

def CNN_LSTM(trainX, trainy,n_steps, n_length):
    model_name = "CNN_LSTM"
    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_name, model


def LSTM_model(trainX, trainy):
    model_name = "LSTM"
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_name, model

def Simple_Bi_LSTM(X_train, Y_train):
    model_name = "SIMPLE_BidirectionalLSTM"
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128,input_shape=[X_train.shape[1], X_train.shape[2]])))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model_name, model
    
def createBidirectionalLSTM(X_train, Y_train):
    model_name = "BidirectionalLSTM"
    n_timesteps, n_features, classNumber = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
    inputs1 = Input(shape=(n_timesteps, n_features))
    bidirect=Bidirectional(LSTM(50,return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(inputs1)
    maxpool=GlobalMaxPool1D()(bidirect)
    dense1 = Dense(50, activation="relu")(maxpool)
    drop = Dropout(0.1)(dense1)
    out = Dense(classNumber, activation="sigmoid")(drop)

    model = Model(inputs=inputs1, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_name, model

def convnet(X_train, Y_train):
    model_name = "2-1DCNN"
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_name, model



""" These models doesn`t work properly """

def multihead_1D_CNN(trainX,trainy):
    model_name = "multihead_1D_CNN"
    
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
     # head 1
    inputs1 = Input(shape=(n_timesteps,n_features))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # head 2
    inputs2 = Input(shape=(n_timesteps,n_features))
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # head 3
    inputs3 = Input(shape=(n_timesteps,n_features))
    conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='multichannel.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_name, model
    
    
def cnnLSTMChollet(max_len, vect_len, vocab_size):
    model_name = 'cnnLSTMChollet'
    model = Sequential()
    model.add(Embedding(vocab_size, vect_len, input_length=max_len))
    model.add(Dropout(0.25))
    model.add(Conv1D(256,
                 5,
                 padding='valid',
                 activation='relu',
                 strides=1))
    model.add(MaxPooling1D(pool_size=3))
    #model.add(LSTM(60))
    model.add(Bidirectional(GRU(60)))#60
    model.add(Dense(50, activation="relu"))#50
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model_name, model


def createCNNLSTMNoEmbeddings2(trainX, trainy):
    model_name = "CNN_BI-LSTM "
    n_steps, n_length = 4, 50 #Timewindow = n_steps*n_length
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model=Sequential()
    #model.add(Embedding(vocab_size, vect_len, input_shape=(max_len, vect_len)))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=8, activation='relu'),input_shape=(n_length, n_features)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(50, dropout=0.1, recurrent_dropout=0.1)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(n_outputs, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_name, model