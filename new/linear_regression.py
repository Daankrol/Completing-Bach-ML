import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.model_selection import KFold, cross_val_score
from methods import SelectionMethod
from process_data import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import itertools
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.utils import _pandas_indexing
from sklearn.utils.extmath import softmax
mpl.use('TkAgg')

def get_standardized_input(inputs, window_size, n_features):

    inputs_array = np.array(inputs)
    inputs_reshaped = np.reshape(inputs_array, (len(inputs_array)*window_size, n_features))
    
    scaler = StandardScaler().fit(inputs_reshaped)

    inputs_standard = scaler.transform(inputs_reshaped)
    inputs_standard = np.reshape(inputs_standard, (inputs_array.shape))

    return scaler, inputs_standard

def standardize_input(latest_input, window_size, scaler,n_features):

    latest_input_array = np.array(latest_input)
    latest_input_reshaped = np.reshape(latest_input_array, (window_size, n_features))

    latest_input_standard = scaler.transform(latest_input_reshaped)
    latest_input_standard = np.reshape(latest_input_standard, (latest_input_array.shape))
    
    return latest_input_standard

def predict_bach():

    VOICE = 0
    selection_method = SelectionMethod.PROB
    window_size = 35

    data, shift_key = extract_features(voice_number=VOICE)
    inputs, outputs = create_inputs_outputs_from_data(data, shift_key, window_size=window_size)

    n_features = len(data[1])
    scaler, inputs_standard = get_standardized_input(inputs, window_size, n_features)

    outputs_values = []
    outputs = np.array(outputs)

    # LogsiticRegression() requires the labels, not OHE vector in training output
    for vector in outputs:
        value = get_shift_from_probability(vector, shift_key, SelectionMethod.TOP, n=1)
        outputs_values.append(value)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    model.fit(inputs_standard, outputs_values)

    print("sklearn key:", model.classes_)
    # this is the key as used by the predict() from sklearn/linearmodel. Extract this such that we can use our ouwn prediciton methods
    shift_key = list(model.classes_) 

    #------------------------------------------

    predictions = [get_voice(VOICE)[-1]]
    for i in range(500):

        latest_input = inputs[-1]
        latest_input = standardize_input(latest_input, window_size, scaler, n_features)

        latest_pitch = predictions[-1]
        
        probs = model.predict_proba([latest_input])[0]
        soft_probs = softmax([probs])[0]

        predicted_shift = get_shift_from_probability(soft_probs, shift_key, method=selection_method, n=3)
        predicted_pitch = add_predicted_value(inputs, latest_pitch, predicted_shift, shift_key)

        predictions.append(predicted_pitch)

    os.remove('supreme_bach.txt')
    with open('supreme_bach.txt', 'a') as file:

        for p in predictions:
            file.write(str(p) + '\n')
        file.close()

    plt.figure(figsize=(10,4))

    plt.clf()
    voice = get_voice(0)
    plt.plot(voice[-500:], 'b-')
    plt.plot(range(len(voice[-500:]), len(voice[-500:])+len(predictions)), predictions, 'r--')
    plt.title("A predicted continuation of the soprano voice")
    plt.ylabel("MIDI note value")
    plt.xlabel("index")
    plt.savefig('linear_window' + str(window_size) + '.png')
    plt.show()

predict_bach()
