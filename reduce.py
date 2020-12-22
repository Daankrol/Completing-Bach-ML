import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import shuffling
import copy


mpl.use('TkAgg')

# read info from file
file_input = open("F.txt", 'r')
sounds = file_input.readlines()

voice_1 = []
voice_2 = []
voice_3 = []
voice_4 = []

# store each line of pitches as a list
for i in range(0, len(sounds)):
    voice_all = sounds[i].split()
    voice_1.append(int(voice_all[0]))
    voice_2.append(int(voice_all[1]))
    voice_3.append(int(voice_all[2]))
    voice_4.append(int(voice_all[3]))

# translate to pitch duration pairs


def reduce_to_pairs(dat):

    pairs = []
    i = 0

    while i < len(dat):
        duration = 1
        while i+1 < len(dat) and dat[i+1] == dat[i]:
            if duration == 16:
                pairs.append([dat[i], duration])
                duration = 1
            else:
                duration += 1
            i += 1
        pairs.append([dat[i], duration])
        i += 1

    return pairs


pairs_4 = reduce_to_pairs(voice_4)


def construct_windows(voice, window_size):
    windows = []

    for i in range(len(voice)):
        if i+window_size + 1 > len(voice) - 1:  # out of bounds
            break

        w = voice[i:i+window_size+1]  # +1 for y-value
        windows.append(w)
    print(np.array(windows)[0:5])
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

# Own 5-FOLD split


def use_manual_split(voice, voice_windows):
    X_train = voice_windows[:2200, :-1]  # 4/5
    X_test = voice_windows[2200:, :-1]  # 1/5
    y_train = voice_windows[:2200, -1]
    y_test = voice_windows[2200:, -1]

    window_model = LinearRegression()
    window_model.fit(X_train, y_train)

    y_predictions = []

    for i in range(2200, 3500):
        last_window = voice_windows[-1]
        new_x = last_window[1:]  # shift one right
        new_y = window_model.predict([new_x])[0]
        y_predictions.append(new_y)
        new_window = np.append(new_x, new_y)
        voice_windows = np.vstack([voice_windows, new_window])

    plt.clf()
    plt.plot(voice, 'b-')
    plt.plot(range(2200, 3500), y_predictions, 'r--')
    plt.savefig('ownFold.png')


def predict_new_data(voice, original_windows):
    model = LinearRegression()
    voice_windows = copy.deepcopy(original_windows)
    X = voice_windows[:, :-1]
    y = voice_windows[:, -1]
    model.fit(X, y)

    for i in range(3000):
        last_window = voice_windows[-1]
        new_x = last_window[1:]  # shift one right
        new_y = model.predict([new_x])[0]
        new_window = np.append(new_x, new_y)
        voice_windows = np.vstack([voice_windows, new_window])

    plt.clf()
    plt.plot(voice_windows[:, -1], 'g-')
    plt.plot(original_windows[:, -1], 'b-')
    plt.savefig('newly_predicted_data.png')


voice_windows = construct_windows(voice_4, 16)
use_manual_split(voice_4, voice_windows)
# predict_new_data(voice_4, voice_windows)
