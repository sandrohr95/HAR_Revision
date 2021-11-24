import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Input
from keras.models import Model
import General.DL_read_data as read_data
import General.Run_models as run
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import time
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling1D, UpSampling1D
from keras.layers.convolutional import Conv1D
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import General.write_results_csv as wr
import General.UTILS as utils
from keras.layers import BatchNormalization
from imblearn.under_sampling import RandomUnderSampler


def encoder_model(input_window):
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(input_window)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling1D(2, padding="same")(x)
    return encoded


def decoder_model(encoded, features=3):
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(encoded)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding="same")(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(features, 3, activation='sigmoid', padding='same')(x)
    return decoded


def classifier_model(encoded):
    flat = Flatten()(encoded)
    mid = Dense(100, activation='relu')(flat)
    mid1 = Dense(100, activation='relu')(mid)
    mid2 = Dense(100, activation='relu')(mid1)
    mid3 = Dense(100, activation='relu')(mid2)
    output = Dense(n_outputs, activation='softmax')(mid3)
    return output


def under_sample(data):
    X = data[["x-axis", "y-axis", "z-axis", "user-id"]]
    Y = data["ActivityEncoded"]
    under = RandomUnderSampler(sampling_strategy="auto")
    X_resample, y_resample = under.fit_sample(X, Y)
    undersample = pd.concat([X_resample, y_resample], axis=1, sort=False)
    return undersample


file_to_save = 'results/CNN_AE_unlabeled.csv'
path = 'Data_to_work'

filename = path + '/WISDM.csv'

# read labeled dataset as pandas Dataframe
df = pd.read_csv(filename)
df = utils.minmax_scaler(df)

df_train = df[df['user-id'] <= 32]
df_test = df[df['user-id'] < 32]

# start computing time
start_computing_time = time.time()

# generate time windows to execute the algorithm by batch
time_window = 200  # this dataset frequency 20hz == 10 seconds
overlap = 20  # 1 second of overlap
epochs, batch_size = 3, 50
X_train, Y_train = read_data.create_windows_and_labels(time_window, overlap, df_train)
X_test, Y_test = read_data.create_windows_and_labels(time_window, overlap, df_test)

# percentage of unlabeled data.
# Use p = 0.2 (optimal) for 20% of unlabeled data
p = 0.2

# read unlabeled data as panda Dataframe
filename_unlabeled = path + '/data_unlabeled_20Hz.csv'
data_unlabeled = pd.read_csv(filename_unlabeled, nrows=int(len(df_train) * p))
data_unlabeled.rename(columns={'x': 'x-axis', 'y': 'y-axis', 'z': 'z-axis'}, inplace=True)

data_unlabeled = data_unlabeled[['x-axis', 'y-axis', 'z-axis']]

# Join train labeled data with unlabeled to train AutoEncoder
df_encode = pd.concat([df_train, data_unlabeled])
# df_encode = utils.minmaxScaler(df_encode)

# generate time windows to train AutoEncoder
X_encoder = read_data.create_windows_without_label(time_window, overlap, df_encode)

timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]

train_X_encoder, test_X_encoder, _, _ = train_test_split(X_encoder, X_encoder,
                                                         test_size=0.2)  # Split train test unlabeled

scores = list()
loss = list()
recall = list()
f1 = list()

# CROSS VALIDATION 10 K-FOLDS
cv = 1
skf = StratifiedShuffleSplit(n_splits=10, random_state=0, test_size=0.1)
for train_index, test_index in skf.split(X_train, Y_train):
    xTrain = X_train[train_index]
    yTrain = Y_train[train_index]
    xTest = X_train[test_index]
    yTest = Y_train[test_index]

    input_window = Input(shape=(train_X_encoder.shape[1], train_X_encoder.shape[2]))

    # autoencoder joinning encoder and decoder
    autoencoder = Model(input_window,
                        decoder_model(encoder_model(input_window)))
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train autoencoder with labeled + unlabeled samples
    history = autoencoder.fit(train_X_encoder, train_X_encoder,
                              epochs=3,
                              batch_size=batch_size,
                              validation_data=(test_X_encoder, test_X_encoder),
                              callbacks=[
                                  keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
                              ])
    #    utils.loss_plot(history)
    #    predict = autoencoder.predict(test_X_encoder)
    #    utils.plot_examples(test_X_encoder, predict)
    #
    encoded = encoder_model(input_window)  # Create encoder model again

    full_model = Model(input_window, classifier_model(encoded))  # Create the final model Encoder + Dense Layers

    # take enconder weights a set the weigths to my model classifier
    for l1, l2 in zip(full_model.layers[0:12], autoencoder.layers[0:12]):
        l1.set_weights(l2.get_weights())

    # avoid retrain encoder part again to keep the weights
    for layer in full_model.layers[0:12]:
        layer.trainable = False

    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # full_model.summary()

    callbacks = [keras.callbacks.ModelCheckpoint(
        "results/best_models/best_model_unlabeled_" + str(p) + "_cv_" + str(cv) + ".h5",
        save_best_only=True,
        monitor="val_loss"
    ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    ]

    # train the classifier model and save the best models with callbacks
    history = full_model.fit(xTrain, yTrain, validation_data=(xTest, yTest),
                             epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # load the best model to evaluate it
    full_model = keras.models.load_model(
        "results/best_models/best_model_unlabeled_" + str(p) + "_cv_" + str(cv) + ".h5")

    # METRICS
    score = full_model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', score[0])
    loss.append(score[0])  # Get loss
    print('Test accuracy:', score[1])
    scores.append(score[1])  # Get accuracy

    y_pred = full_model.predict(X_test, verbose=1)  # Predict Results

    # run confusion matrix
    run.predict_confusion_matrix(y_pred, X_test, Y_test, Y_train,
                                 "results/confusion_matrix/CNN_AE_unlabeled_" + str(p) + "_cv_" + str(
                                     cv) + ".eps")

    Y_test_score = Y_test.argmax(axis=1)
    y_pred_score = y_pred.argmax(axis=1)

    # get f1-score
    f1_macro = f1_score(Y_test_score, y_pred_score, average='macro')
    f1_weighted = f1_score(Y_test_score, y_pred_score, average='weighted')
    print('f1_macro:', f1_macro)
    print('f1_weighted:', f1_weighted)
    f1.append(f1_weighted)

    # get recall
    recall_macro = recall_score(Y_test_score, y_pred_score, average='macro')
    recall_weighted = recall_score(Y_test_score, y_pred_score, average='weighted')
    print('recall_macro:', recall_macro)
    print('recall_weighted:', recall_weighted)
    recall.append(recall_weighted)

    # plot the model's training and validation loss
    utils.accuracy_plot(history, p, cv)

    # write results of each cross validation in csv
    wr.write_results(file_to_save, start_computing_time, "CNN_AE_UNLABELED_CV_" + str(cv), time_window, overlap, epochs,
                     batch_size, score[1], score[0], recall_weighted, f1_weighted, p)
    cv += 1

    # clear keras session to train another model
    keras.backend.clear_session()

    # Use break to run cross validation only once
    break

# SUMMARIZE RESULTS
print("RESULTADOS FINALES PARA UNLABELED: " + str(p))
scores_mean = np.mean(scores)
print("Accuracy: " + str(scores_mean))
loss_mean = np.mean(loss)
print("loss: " + str(loss_mean))
recall_mean = np.mean(recall)
print("recall: " + str(recall_mean))
f1_mean = np.mean(f1)
print("f1_score: " + str(f1_mean))

# write final results in csv
wr.write_results(file_to_save, start_computing_time, "SUMMARY", time_window, overlap, epochs, batch_size, scores_mean,
                 loss_mean,
                 recall_mean, f1_mean, p)

#    RETRAIN THE FULL MODEL AGAIN
#    for layer in full_model.layers[0:4]:
#        layer.trainable = True
#
#    full_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#    history = full_model.fit(X_train, Y_train,validation_data=(X_test, Y_test),
#                  epochs=epochs, batch_size=batch_size,callbacks=callbacks, verbose=1)
