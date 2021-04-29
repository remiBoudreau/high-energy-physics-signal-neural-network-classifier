# Filename: hep_nn.py
# Dependencies: hep_nn_utilities, keras, matplotlib.pyplot, pandas, random, sklearn, tensorflow
# Author: Jean-Michel Boudreau
# Date: May 1, 2019

'''
Loads HEPMASS data set (must be colocated in directory with this script) and 
trains a deep neural network in the standard "funnnel-like" architecture to 
predict whether a given event is a signal or background noise.
'''

# import libraries
from hep_nn_utilities import load_hep_data, process_data, get_model, loss_plot
from keras import optimizers
from keras.callbacks import EarlyStopping

# load the HEPMASS data set
print('Reading HEP file.')
hep_data = load_hep_data()
# process the data and split into feature and target training/testing subsets
print('Processing HEPMASS data')
X_train, X_test, y_train, y_test = process_data(hep_data)
# define DNN architecture to be used. Follows standard "funnel" like arch.
# changing numbers of hidden layers does not change model output sig
print('Building network.')
model = get_model(14, 7)
# define optimizer as Nesterov momentum SGD. Nesterov momentum SGD chosen over
# Adam optimizer because Adam optimizer has been shown to lead to solutions
# that generalize poorly on some datasets. See reference [1].
sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
# construct dnn
model.compile(optimizer = sgd, metrics = ['accuracy'], 
              loss = 'binary_crossentropy')
# statement for early stopping callback to prevent overfitting and save time
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# train dnn using mini-batch gradient descent and use a validation set which 
# is 5% the size of the training set. 
print('Training network.')
hep_dnn = model.fit(X_train, y_train, validation_split=0.05, epochs=1000, 
                batch_size=100, callbacks=[es])
# get accuracy of model on test set
score = model.evaluate(X_test, y_test)
print('The test score is '+ str(score))
# graph the loss ca. from the logistic regression cost function (identical to 
# cross-entropy cost function with two classes) as a function of epoch
loss_plot(hep_dnn)

'''
Links to references:
    [1] https://arxiv.org/pdf/1705.08292.pdf
    [2] https://arxiv.org/pdf/1502.03167.pdf
    [3] https://arxiv.org/pdf/1801.05134.pdf
'''
