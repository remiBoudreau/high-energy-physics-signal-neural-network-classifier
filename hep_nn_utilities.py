# Filename: hep_nn_utilities.py
# Dependencies: keras, matplotlib.pyplot, pandas, random, sklearn, tensorflow
# Author: Jean-Michel Boudreau
# Date: May 1, 2019

'''
Module containing all function required by hep_nn.py driver.
'''

# import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from random import sample
import keras.models as km
import keras.layers as kl
import matplotlib.pyplot as plt

'''
Loads the HEPMASS data set in csv format. File must be colocated with script.
Returns the HEPMASS data as a pandas dataframe.
'''
def load_hep_data():
	file_name = "HEPMASS_small.txt"
	data = pd.read_csv(file_name)
    
	return data

'''
Processes the data (required as argument with type as pandas dataframe) for 
feeding into the deep neural network: 
    1. Scales the features in order to _____ . 
    2. Randomly samples the data without replacement in order to create
       training, validation and testing subsets.
Returns training and testing data subsets for both the features
(labelled as 'X') and the target(labelled as 'y').
       
'''
def process_data(data):
    # feature scaling to avoid bias in a single/few feature that are large in
    # magnitude as well as speed up convergence (when using any GD methods). 
    # See reference [2]
    scaler = MinMaxScaler()
    data = data.drop(['mass'], axis=1)
    column_list = list(data)
    data[column_list] = scaler.fit_transform(data)
    # randomly sample (to avoid bias) 20% of the whole data set for testing; 
    # train on reminaing 80%. 
    n_event = len(data.index)
    data_in = range(n_event)
    n_test = int(0.2*n_event)
    test_in = sample(data_in, n_test)
    train_in = list(set(data_in) - set(test_in)) 
    data_test = data.iloc[test_in]
    data_train = data.iloc[train_in]    
    # split data into features (inputs) and targets (outputs)
    target = "# label"                              
    feature_list = list(data.drop([target], axis=1))
    X_test = data_test[feature_list].values
    y_test = data_test[target].values
    X_train = data_train[feature_list].values
    y_train = data_train[target].values

    return X_train, X_test, y_train, y_test

'''
Creates a deep neural network with 2 hidden layers with n_hidden1 nodes in the
first hidden layer and n_hidden2 nodes in the second layer and carries out
binary classification using logistic regression.
'''
def get_model(n_hidden1, n_hidden2):
    # construct input layer
    visible = kl.Input(shape=(27,), name='input')
    
    # construct 1st hidden layer
    hidden1 = kl.Dense(n_hidden1,
                       # He initialization does not improve model
                       # kernel_initializer='he_uniform',
                       name='hidden1')(visible)
    # BN allows model to converge faster and improves upon model by ~0.2 - 0.4%
    hidden1 = kl.BatchNormalization(name='hidden1_batch_normalization')(hidden1)
    # model w/ elu improves upon model w/ relu by ~0.4%
    hidden1 = kl.Activation('elu', name='hidden1_activation')(hidden1)
    # Adding dropout does not improve model; should not be used with batch
    # normalization. Refer to reference [3].
    # hidden1 = kl.Dropout(0.5)(hidden1) 
    
    # construct 2nd hidden layer
    hidden2 = kl.Dense(n_hidden2, 
                       # He initialization does not improve model
                       # kernel_initializer = 'he_uniform',
                       name='hidden2')(hidden1)
    # BN allows model to converge faster and improves upon model by ~0.2 - 0.4%
    hidden2 = kl.BatchNormalization(name='hidden2_batch_normalization')(hidden2)
    # model w/ elu improves upon model w/ relu by ~0.4%
    hidden2 = kl.Activation('elu', name='hidden2_activation')(hidden2)
    # Adding dropout does not improve model; should not be used with batch
    # normalization. Refer to reference [3].
    # hidden2 = kl.Dropout(0.5)(hidden2)

    # construct output layer
    y_pred = kl.Dense(1, name = 'output', 
                       activation = 'sigmoid')(hidden2)
    # construct dnn architecture
    model = km.Model(inputs = visible, outputs = y_pred)

    return model

'''
Creates plot of the loss (specified by the cost function) as a fn of epoch.
'''
def loss_plot(dnn_training):
    plt.plot(dnn_training.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('HEPMASS_small_loss.png')
