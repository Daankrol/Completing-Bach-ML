import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from methods import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import itertools
import os

from sklearn.utils import _pandas_indexing
from read_data import add_predicted_value, get_input_output, get_pitch_duration_pairs, get_pitch_features, get_pitch_from_probability, get_voice
mpl.use('TkAgg')


def predict_bach():
    method = WindowMethod.CUMULATIVE
    prob_method = ProbabilityMethods.VALUES
    selection_method = SelectionMethod.PROB
    inputs, outputs, key = get_input_output(
        voice_number=0, method=method, prob_method=prob_method, window_size=4, use_features=True)

    # input_windows = inputs[:2200]
    # teacher_values = outputs[:2200]

    polynomial_features = PolynomialFeatures(degree = 2)
    inputs_TRANSF = polynomial_features.fit_transform(inputs)

    model = LinearRegression()
    model.fit(inputs_TRANSF, outputs)

    # for x in range(2200, 2400):
    predictions = []
    for x in range(1000):

        latest_input = [inputs[-1]]
        latest_input_TRANSF = polynomial_features.fit_transform(latest_input)

        probs = model.predict(latest_input_TRANSF)[0]

        predicted_pitch = get_pitch_from_probability(
            probs, key, method=selection_method)
        predictions.append(predicted_pitch)

        add_predicted_value(inputs, predicted_pitch,
                            method=method, use_features=True)
    os.remove('supreme_bach.txt')
    with open('supreme_bach.txt', 'a') as file:

        for p in predictions:
            file.write(str(p) + '\n')
        file.close()

    plt.clf()
    voice = get_voice(0)
    plt.plot(voice, 'b-')
    plt.plot(range(len(voice), len(voice)+len(predictions)), predictions, 'r--')
    plt.savefig('mulivariate_polynomial_regression_' + str(method).split('.')[1] + '.png')
    plt.show()


predict_bach()


def construct_windows(pd_pairs, window_size):
    windows = []

    for i in range(len(pd_pairs)):
        if i+window_size + 1 > len(pd_pairs) - 1:  # out of bounds
            break
        window = []
        for j in range(window_size + 1):
            window.append(pd_pairs[i+j][0])
            window.append(pd_pairs[i+j][1])
        # add Y value
        window.append(pd_pairs[i+window_size+1][0])  # [0] for pitch
        windows.append(window)
    return np.array(windows)


def use_kfold(voice, voice_windows):
    X = voice_windows[:, :-1]
    y = voice_windows[:, -1]

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        window_model = LinearRegression()
        window_model.fit(X_train, y_train)
        y_predictions = window_model.predict(X_test)
        print(len(X_train), len(y_predictions))
        break
    plt.clf()
    plt.plot(voice, 'b-')
    plt.plot(y_predictions, 'r--')
    plt.savefig('Kfold_ln_window_predict.png')


def use_manual_split(voice, windows):
    X_train = windows[:, :-1]  # 4/5
    X_test = windows[2200:, :-1]  # 1/5
    y_train = windows[:, -1]
    y_test = windows[2200:, -1]

    window_model = LinearRegression()
    window_model.fit(X_train, y_train)

    y_predictions = []
    windows = windows[:, :-1]  # remove all y values

    predicted_windows = [X_train[-1]]
    for i in range(1000):
        last_window = predicted_windows[-1]
        # window = [p1, d1, p2, d2]

        # shift right * 2 => window = [p2,d2, y , _, _]
        # check if y == p2, then y = p2, d_new = d2+1
        # else y = p_new,  d_new = 1
        # new_window = [p2,d2,p3,d3,y]
        pitch = round(window_model.predict([last_window])[0])
        y_predictions.append(pitch)
        previous_pitch, previous_duration = last_window[-2], last_window[-1]

        duration = 1
        # FIXME pitch prediction is a floating point value so this is never equal.
        if pitch == previous_pitch:
            duration = previous_duration + 1

        new_window = last_window[2:]  # shift right, removing first p,d pair
        new_window = np.append(new_window, pitch)
        new_window = np.append(new_window, duration)
        predicted_windows = np.vstack([predicted_windows, new_window])

    print(y_predictions[0:100])
    plt.clf()
    plt.plot(voice, 'b-')
    plt.plot(range(len(voice), len(voice)+1000), y_predictions, 'r--')
    plt.savefig('ownFold.png')


# voice = get_voice(voice_number=3)
# pd_pairs = get_pitch_duration_pairs(voice_number=3, method='shift')
# windows = construct_windows(pd_pairs, 16)

# use_manual_split(voice, windows)


# 5 keys
# key 1 = [1, 0,0,0,0]
# [0, 43, 44, 45]
