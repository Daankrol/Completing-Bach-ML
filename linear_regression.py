import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import itertools

from sklearn.utils import _pandas_indexing
from read_data import get_input_output, get_pitch_duration_pairs, get_pitch_features, get_voice
mpl.use('TkAgg')


def predict_bach():
    method = 'cumulative'
    prob_method = 'values'
    inputs, outputs, key = get_input_output(
        voice_number=0, method=method, prob_method=prob_method, window_size=4, use_features=True)
    pitches = inputs[0][:2200]
    durations = inputs[1][:2200]
    teacher_values = outputs[:2200]
    pitch_features = []
    
    for i in inputs:
        print(i)
    exit()

    for window in pitches:
        # w_features = []
        # for p in window:
        #     w_features.append()
        window_features = [get_pitch_features(p) for p in window]
        window_features = list(itertools.chain.from_iterable(window_features))
        pitch_features.append(window_features)

    print('pf', pitch_features[0])
    print('d', durations[0])

    pd_windows = []
    for i in range(len(pitches)):
        pd_windows.append(pitch_features[i] + durations[i])

    print('\n\n\n', pd_windows[0])
    # exit()

    model = LinearRegression()
    model.fit(pd_windows, teacher_values)

    # for x in range(2200, 2400):
    ps = []
    for x in range(2200):
        probs = model.predict([pd_windows[x]])[0] * 100
        ps.append(key[np.where(probs == max(probs))][0])

    with open('supreme_bach.txt', 'a') as file:
        for p in ps:
            file.write(str(p) + '\n')
        file.close()

    # plt.clf()
    # voice = get_voice(0)
    # plt.plot(voice[:2200], 'b-')
    # plt.plot(ps[:2200], 'r--')
    # plt.savefig('mulivariate_linear_regression_' + method + '.png')
    # plt.show()


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
