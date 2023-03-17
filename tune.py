%%time
##############################
##          IMPORTS         ##
##############################
# THIRD PARTY IMPORTS
# For arrays
import numpy as np
# For importing the data set
import pandas as pd
# For graphing
import matplotlib.pyplot as plt
# For calculating the r2 score
from sklearn.metrics import r2_score
# For imnporting files from other directories
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data_dir = 'Data/'

# LOCAL IMPORTS
# Local imports stored in this folder
# For graphing functionality and error functions
from Analysis import *
# To generates the data file paths
from DataImportAndFormat import *
# To perform the bayesian extrapolation
from BayesianExtrapolation import *
from Support import *

start_dim = 5
dim = 21
seq = 3

filenames, numbers = generate_filenames (data_dir)

def setup_nn (seq, neurons, hidden_activation, output_activation):
    # define the keras model
    model = Sequential()
    model.add(Dense(neurons, input_dim=seq, activation=hidden_activation))
    model.add(Dense(1, activation=output_activation))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam')
    return model
    
####################################
## SEQUENTIAL EXTRAPOLATE SKLEARN ##
####################################
def sequential_extrapolate_keras(R, y_train, num_points, seq):
    """
        Inputs:
            R (an object): A trained Scikit-Learn regression model
            y_train (a list): the y component of the training data set, unformatted
            num_points (an int): the number of points to be in the extrapolated data set
            seq (an int): the SRE length of sequence that R was trained with
        Returns:
            y_test (a list): the extrapolated data set
            y_std (a list): the uncertainity on each point in the extrapoalted set
        Performs SRE using a trained Scikit-Learn regression model.
    """
    # Make sure inputs are of the proper type
    assert isinstance(num_points, int)
    assert isinstance(seq, int)

    # Add the training data to the extrapolated data set and no uncertainities for the
    # training data
    y_test = y_train.copy().tolist()

    # Extrapolate until enough data points have been predicted
    while len(y_test) < num_points:
        next_test = np.array(y_test[-seq:])
        next_test = next_test.reshape(1,-1)
        point= R.predict(next_test,verbose=None)
        y_test.append(point[0][0])


    # Return the predicted data set and uncertainities
    return y_test
    
%%time
best_err = 100
best_params = []
best_preds = []
best_std = []
for num_neurons in np.arange(1,16):
    for hidden_activation in ['relu', 'tanh', 'sigmoid', None]:
        for output_activation in ['relu', 'tanh', 'sigmoid', None]:
            for epochs in np.arange(1,11):
                iterations = 100
                err = 0
                saved_preds = []
                saved_std = []
                for filename in filenames:
                    states, mbpt, cc, mbpt_times, cc_times = import_and_split_columns (filename,sep=' ')
                    training_data = cc[start_dim:dim]/mbpt[start_dim:dim]
                    x_train, y_train = format_sequential_data (training_data, seq)
                    x_train = np.asarray(x_train)
                    x_train = x_train.reshape((len(x_train), -1))
                    y_train = np.asarray(y_train)
                    predictions = []
                    for i in range(iterations):
                        nn = setup_nn (seq, num_neurons, hidden_activation, output_activation)
                        nn.fit(x_train, y_train, epochs=epochs, verbose=None)
                        y_test = sequential_extrapolate_keras(nn, training_data, 50, seq)
                        predictions.append(y_test[-1])
                    print(predictions)
                    print(get_70(mbpt))
                    cc_predictions = get_70(mbpt)*np.asarray(predictions)
                    average_cc_prediction = np.average(cc_predictions)
                    std_cc_prediction = np.std(cc_predictions)
                    err.append(np.abs(average_cc_prediction-get_70(cc)))
                    saved_preds.append(average_cc_prediction)
                    saved_std.append(std_cc_prediction)
                average_err = np.average(err)
                if err < best_err:
                    best_params = [num_neurons, hidden_activation, output_activation, epochs]
                    best_preds = saved_preds
                    best_std = saved_std 

print("BEST PARAMS")
print(best_params)
print("BEST PREDICTIONS")
print(best_preds)
print("BEST STD")
print(best_std)    
