import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.model_selection import KFold, cross_val_score
from methods import *
from process_data import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import itertools
import os
from sklearn.utils import _pandas_indexing 
mpl.use('TkAgg')

def predict_bach():

    VOICE = 0
    selection_method = SelectionMethod.PROB
    window_size=30

    data, shift_key = extract_features(voice_number=VOICE)
    inputs, outputs = create_inputs_outputs_from_data(data, shift_key, window_size=window_size)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(inputs, outputs)
        
    predictions = []
    for i in range(1000):

        latest_input = inputs[-1]
        latest_pitch = predictions[-1] if i>0 else get_voice(VOICE)[-1]
        
        probs = model.predict([latest_input])[0]

        predicted_shift = get_shift_from_probability(probs, shift_key, method=selection_method)
        predicted_pitch = add_predicted_value(inputs, latest_pitch, predicted_shift, shift_key)

        predictions.append(predicted_pitch)

    os.remove('supreme_bach.txt')
    with open('supreme_bach.txt', 'a') as file:

        for p in predictions:
            file.write(str(p) + '\n')
        file.close()

    plt.clf()
    voice = get_voice(0)
    plt.plot(voice, 'b-')
    plt.plot(range(len(voice), len(voice)+len(predictions)), predictions, 'r--')
    plt.savefig('linear_window' + str(window_size) + '.png')
    plt.show()

predict_bach()