import numpy as np
import random
import itertools
from methods import SelectionMethod

def get_voice(voice_number=0, remove_zeros=True):
    """
    Get voice
    :param voice_number: default: 0
    :param remove_zeros default: True
    :return:
    """

    file_input = open("F.txt", 'r')
    sounds = file_input.readlines()

    voice = []
    for i in range(0, len(sounds)):
        voice_all = sounds[i].split()
        voice.append(int(voice_all[voice_number]))

    if remove_zeros and voice:
        voice = [pitch for pitch in voice if pitch!=0]

    return voice


def get_shift_key(voice):
    """
    Get key to represent pitch step differences
    :param voice:
    :return:
    """
    shift_key = []
    for i in range(len(voice)-1):
        shift = voice[i+1] - voice[i]
        shift_key.append(shift)

    shift_key = list(set(shift_key))

    return shift_key


def extract_features(voice_number=0):
    """
    Process pitch values to features and a one hot encoding of its shift relative the preceding pitch value
    :param voice_number 0t/m3
    :return: data, shift_key
    """
    voice = get_voice(voice_number)
    shift_key = get_shift_key(voice)

    processed_data = []
    
    for i in range(1,len(voice)):
        current, previous = voice[i], voice[i-1]
        pitch_features = get_pitch_features(current, voice=voice_number)
        shift_one_hot = [0] * len(shift_key)
        pitch_shift = current - previous
        shift_one_hot[shift_key.index(pitch_shift)] = 1 #set one hod encoing vector for the pitch shift

        processed_data.append(pitch_features + shift_one_hot)

    return processed_data, shift_key


def create_inputs_outputs_from_data(data, shift_key, window_size=44):
    """
    Generate inputs and outputs for a certain window_size
    :param data: dataset created with "extract_features" function
    :param shift_key:
    :param window_size: default: 44
    :return: inputs, outputs
    """
    inputs= []
    outputs= []
    one_hot_length = len(shift_key) # to separate input from output

    for i in range(window_size, len(data)):
        
        temp_input = []
        for j in range(window_size, 0, -1): #flattening method

            temp_input = temp_input + data[i-j]

        # temp_array = np.array(data[i-window_size:i])
        # flat_array = temp_array.flatten()
        inputs.append(temp_input)
        outputs.append(data[i][-one_hot_length:])

    return inputs, outputs 


def get_shift_from_probability(prob, key, method=SelectionMethod.HIGHEST, n=3):
    """
    Select a pitch value given a probability vector
    :param prob: the key given during i/o creation
    :param output: the original teacher values
    :param select_method "highest", "topN", "weighted", "random", "probN"
    :return: predicted_shift
    """
    if method == SelectionMethod.TOP:  # equal chance for top n
        top_n = []
        if n > len(key):
            exit('ERROR: Selecting top n values but n is larger than number of possible outcomes. Select n<=%d' % len(key))
        for i in range(n):
            idx = random.choice(np.where(prob == max(prob))[0])
            # idx = idx if type()
            top_n.append(key[idx])
            prob[idx] = 0

        predicted = random.choice(top_n)

    if method == SelectionMethod.PROB:  # weighted chance for top n
        top_n = []
        top_prob = []
        if n > len(key):
            exit('ERROR: Selecting top n values but n is larger than number of possible outcomes. Select n<=%d' % len(key))
        for i in range(n):
            idx = random.choice(np.where(prob == max(prob))[0])
            # idx = idx if type()
            top_n.append(key[idx])
            top_prob.append(max(prob))
            prob[idx] = 0

        idx = random.choices(range(len(top_prob)), weights=top_prob)[0]
        predicted = top_n[idx]

    if method == SelectionMethod.WEIGHTED:
        idx = random.choices(range(len(prob)), weights=prob)[0]
        predicted = key[idx]

    if method == SelectionMethod.HIGHEST:
        idx = np.where(prob == max(prob))
        print("idx:", idx)
        predicted = key[idx]
        print("predicted:", predicted)
        if type(predicted) != int:  # in case there are two probabilities with equal value
            predicted = random.choice(predicted)

    if method == SelectionMethod.RANDOM:
        predicted = random.choice(key)

    return predicted


def add_predicted_value(inputs, latest_pitch, shift, shift_key, voice):
    """
    Add predicted shift and create new window for next iteration
    :param inputs:
    :param latest_pitch: previous pitch value (absolute/actual value, not the shift)
    :param shift: shift value predicted from model output
    :param shift_key:
    :return: predicted_pitch
    """

    old_input = inputs[-1] #latest input window
    one_hot_length = len(shift_key)

    predicted_pitch = latest_pitch + shift

    pitch_features = get_pitch_features(predicted_pitch, voice)
    shift_one_hot = [0] * one_hot_length
    shift_one_hot[shift_key.index(shift)] = 1

    new_data_entry = pitch_features + shift_one_hot

    #remove the oldest data point in the most recent window and add the prediction accordingly
    new_window = old_input[len(new_data_entry):] + new_data_entry
    inputs.append(new_window)

    return predicted_pitch


def get_pitch_features(midi_note, voice):
    """
    Returns thelog pitch, x,y coordinate of the croma circle and x,y coordinate for the circle of fifths
    :param midi_note:
    :return:
    """
    chroma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    radius_chroma = 1
    c5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
    radius_c5 = 1

    # 53 translates to D-minor (F-major) in the chromatic circle
    note = (midi_note - 53) % 12

    chroma_angle = (chroma[note]) * (360 / 12)
    c5_angle = (c5[note]) * (360 / 12)

    log_pitch = get_log_pitch(midi_note, voice)
    chroma_x = radius_chroma * np.sin(np.deg2rad(chroma_angle))
    chroma_y = radius_chroma * np.cos(np.deg2rad(chroma_angle))
    c5_x = radius_c5 * np.sin(np.deg2rad(c5_angle))
    c5_y = radius_c5 * np.cos(np.deg2rad(c5_angle))

    return [log_pitch, chroma_x, chroma_y, c5_x, c5_y]


def get_log_pitch(midi_note, voice):
    """
    Returning the log_pitch of a midi note
    :param midi_note:
    :return:
    """
    max_notes = [76, 71, 62, 54]  # per voice number
    min_notes = [54,45,40,28]
    n = midi_note - 69  # 69 is the midi of A over middle C
    fx = pow(2, (n / 12)) * 440  # 220 is the frequency of A over middle C
    max_note = max_notes[voice]
    min_note = 0 
    min_p = 2 * np.log2(pow(2, ((min_note - 69) / 12)) * 440)
    max_p = 2 * np.log2(pow(2, ((max_note - 69) / 12)) * 440)
    log_pitch = 2 * np.log2(fx) - max_p + (max_p - min_p) / 2
    return log_pitch   