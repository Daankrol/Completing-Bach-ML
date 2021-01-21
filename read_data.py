import numpy as np
import random
import itertools
from methods import WindowMethod, ProbabilityMethods, SelectionMethod


def get_voice(voice_number):
    """
    Get voice
    :param voice_number:
    :return:
    """
    file_input = open("F.txt", 'r')
    sounds = file_input.readlines()

    voice = []
    for i in range(0, len(sounds)):
        voice_all = sounds[i].split()
        voice.append(int(voice_all[voice_number]))

    return voice


def get_input_output(voice_number=0, method=WindowMethod.CUMULATIVE, prob_method=None, window_size=16, use_features=False):
    """
    :param voice_number: 0 t/m 3
    :param method: 'cumulative' or 'shift'
    :param prob_method: provide output as probability vector. None, 'range', 'values'
    :param window_size:
    """
    pairs = get_pitch_duration_pairs(voice_number=voice_number, method=method)

    pitch_array = [p[0] for p in pairs]
    duration_array = [p[1] for p in pairs]

    # pitch_input = []
    # duration_input = []

    inputs = []
    output = []

    if method == WindowMethod.SHIFT:
        # no additional function to spare looping multiple times
        for i in range(len(pairs)):
            if i+window_size > len(pairs) - 1:  # out of bounds
                break
            p = pitch_array[i:i+window_size]
            d = duration_array[i:i+window_size]
            o = pitch_array[i+window_size]

            empty = []
            if use_features:
                pf = [get_pitch_features(x) for x in p]
            for w in range(window_size):
                empty.append(p[w])
                if use_features:
                    empty = empty + pf[w]
                empty.append(d[w])

            inputs.append(empty)
            output.append(o)

    elif method == WindowMethod.CUMULATIVE:
        flag = 0
        for i in range(window_size-1, len(pairs)):
            # TODO: this is not perfect but the problem scenario dow not occur in our data, this is probbibly not the preferred method anyway
            if duration_array[i] == 1 and flag == 0:
                flag = 1
            else:
                flag = 1
                for j in range(1, duration_array[i]+1):
                    p = pitch_array[i-window_size+1:i+1]
                    d = duration_array[i-window_size+1:i]+[j]
                    o = pitch_array[i]

                    empty = []
                    if use_features:
                        pf = [get_pitch_features(x) for x in p]
                    for w in range(window_size):
                        empty.append(p[w])
                        if use_features:
                            empty = empty + pf[w]
                        empty.append(d[w])

                    inputs.append(empty)
                    output.append(o)

    key = []  # later used for making a prediciton from the probability vector

    if prob_method != None:
        output = np.array(output)
        if prob_method == ProbabilityMethods.RANGE:  # use entire range of pitch
            temp = np.where(output == 0)
            nonzero_min = min(np.delete(output, temp))
            # +1 for inclusion of max and for zero tone
            output_prob = np.zeros(
                (len(output), (max(output) - nonzero_min + 1 + 1)), dtype=int)
            for i in range(len(output)):
                if output[i] == 0:
                    output_prob[i][0] = 1
                else:
                    output_prob[i][output[i]-nonzero_min+1] = 1

            key = np.append([0], np.arange(nonzero_min, max(output) + 1, 1))

        elif prob_method == ProbabilityMethods.VALUES:  # use only before seen pitch values
            pitch_unique = np.unique(output)
            output_prob = np.zeros((len(output), len(pitch_unique)), dtype=int)
            for i in range(len(output)):
                output_prob[i][np.where(pitch_unique == output[i])] = 1

            key = pitch_unique

        output = output_prob

    # inputs = [pitch_input, duration_input]

    return inputs, output, key


def get_pitch_from_probability(prob, key, method=SelectionMethod.HIGHEST, n=3):
    """
    Select a pitch value given a probability vector
    :param prob: the key given during i/o creation
    :param output: the original teacher values
    :param select_method "highest", "topN", "weighted", "random", "probN"
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

        # print("top:", top_n)
        # print('top prob:', top_prob)

        idx = random.choices(range(len(top_prob)), weights=top_prob)[0]
        predicted = top_n[idx]

        # print(predicted)

    if method == SelectionMethod.WEIGHTED:
        idx = random.choices(range(len(prob)), weights=prob)[0]
        predicted = key[idx]

    if method == SelectionMethod.HIGHEST:
        idx = np.where(prob == max(prob))
        predicted = key[idx]
        if type(predicted) != int:  # in case there are two probabilities with equal value
            predicted = random.choice(predicted)

    if method == SelectionMethod.RANDOM:
        predicted = random.choice(key)

    return predicted


def add_predicted_value_old(inputs, predicted, method=WindowMethod.CUMULATIVE):
    """
    Add the newly predicted pitch value to the input data
    :param inputs:
    :param predicted:
    :param method: 
    """
    if method == WindowMethod.SHIFT:
        # shift window and append predicted value
        inputs[0].append(inputs[0][-1][1:]+[predicted])
        if predicted == inputs[0][-1][-2]:  # predicted is same as previous pitch
            prev_dur = inputs[1][-1][-1]
            # shift window and append duration value that is an increment of the d at t-1
            inputs[1].append(inputs[1][-1][1:]+[prev_dur+1])
        else:  # predicted not the same as previous pitch
            # shift window and append duration of 1
            inputs[1].append(inputs[1][-1][1:]+[1])

    if method == WindowMethod.CUMULATIVE:
        if predicted == inputs[0][-1][-1]:  # predicted is same as previous pitch
            # window shift results in the same pitch input
            inputs[0].append(inputs[0][-1])
            prev_dur = inputs[1][-1][-1]
            # window shift results in an increment of the duration value at the d at t-1
            inputs[1].append(inputs[1][-1][:-1]+[prev_dur+1])
        else:  # predicted not the same as previous pitch
            # shift window and append predicted value
            inputs[0].append(inputs[0][-1][1:]+[predicted])
            # shift window and append duration of 1
            inputs[1].append(inputs[1][-1][1:]+[1])


def add_predicted_value(inputs, predicted, method=WindowMethod.CUMULATIVE, use_features=False):
    """
    Add the newly predicted pitch value to the input data
    :param inputs:
    :param predicted:
    :param method: 
    :param use_features:
    """
    old_input = inputs[-1]

    # print(old_input)
    # print('here:',old_input[2:])
    # print(old_pitch)
    # print(old_duration)

    if use_features == False:  # only pitch and duration
        old_pitch = inputs[-1][::2]  # pitch at even indices
        old_duration = inputs[-1][1::2]  # duration at odd indices
        if method == WindowMethod.SHIFT:
            # predicted is same as previous pitch
            if predicted == old_pitch[-1]:
                new_duration = old_duration[-1] + 1
                inputs.append(
                    np.append(old_input[2:], np.append([predicted], [new_duration])))
            else:
                inputs.append(
                    np.append(old_input[2:], np.append([predicted], [1])))

        if method == WindowMethod.CUMULATIVE:
            # predicted is same as previous pitch
            if predicted == old_pitch[-1]:
                new_duration = old_duration[-1] + 1
                inputs.append(np.append(old_input[:-1], [new_duration]))
            else:
                inputs.append(
                    np.append(old_input[2:], np.append([predicted], [1])))
    else:  # extensive features
        old_pitch = inputs[-1][::7]
        old_duration = inputs[-1][6::7]  # duration at odd indices
        pf = get_pitch_features(predicted)  # pitch feature array
        if method == WindowMethod.SHIFT:
            # predicted is same as previous pitch
            if predicted == old_pitch[-1]:
                new_duration = old_duration[-1] + 1
                inputs.append(old_input[7:] +
                              [predicted] + pf + [new_duration])
            else:
                inputs.append(old_input[7:] + [predicted] + pf + [1])
        if method == WindowMethod.CUMULATIVE:
            # predicted is same as previous pitch
            if predicted == old_pitch[-1]:
                new_duration = old_duration[-1] + 1
                inputs.append(old_input[:-1] + [new_duration])
            else:
                inputs.append(old_input[7:] + [predicted] + pf + [1])


def get_pitch_duration_pairs(voice_number=0, method=WindowMethod.CUMULATIVE):
    """
    Translate to additive pitch duration pairs
    :param method: 'cumulative' or 'shift'
    :return:
    :param voice_number:
    :return:
    """
    voice = get_voice(voice_number=voice_number)
    pairs = []
    i = 0
    while i < len(voice):
        duration = 1
        while i + 1 < len(voice) and voice[i + 1] == voice[i]:
            if method == WindowMethod.SHIFT:
                pairs.append([voice[i], duration])
            duration += 1
            i += 1
        pairs.append([voice[i], duration])
        i += 1
    return pairs


def get_pitch_features(midi_note):
    """
    Returns the chroma
    :param midi_note:
    """
    chroma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    radius_chroma = 1
    c5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
    radius_c5 = 1

    # 53 translates to D-minor (F-major) in the chromatic circle
    note = (midi_note - 53) % 12

    chroma_angle = (chroma[note]) * (360 / 12)
    c5_angle = (c5[note]) * (360 / 12)

    log_pitch = get_log_pitch(midi_note)
    chroma_x = radius_chroma * np.sin(np.deg2rad(chroma_angle))
    chroma_y = radius_chroma * np.cos(np.deg2rad(chroma_angle))
    c5_x = radius_c5 * np.sin(np.deg2rad(c5_angle))
    c5_y = radius_c5 * np.cos(np.deg2rad(c5_angle))

    return [log_pitch, chroma_x, chroma_y, c5_x, c5_y]


def get_log_pitch(midi_note):
    """
    Returning the log_pitch of a midi note
    :param midi_note:
    :return:
    """
    n = midi_note - 69  # 69 is the midi of A over middle C
    fx = pow(2, (n / 12)) * 440  # 220 is the frequency of A over middle C
    min_note = 0  # FIXME fixed for voice 0
    max_note = 76  # FIXME fixed for voice 0
    min_p = 2 * np.log2(pow(2, ((min_note - 69) / 12)) * 440)
    max_p = 2 * np.log2(pow(2, ((max_note - 69) / 12)) * 440)
    log_pitch = 2 * np.log2(fx) - max_p + (max_p - min_p) / 2
    return log_pitch


# inputs, output, key = get_input_output(
#     voice_number=3, method='shift',  prob_method='range', window_size=2)

# print('input:', inputs[-2:])

# prob = np.zeros(len(key))
# prob[0] = 0.5
# # prob[6]=.25
# # prob[7]=.25
# print('inputs[-1]:', inputs[-1])

# predicted = 66
# predicted = get_pitch_from_probability(prob, key, method='highest')
# print('predicted:', predicted)

# add_predicted_value(inputs, predicted, method='shift')
# print('input:', inputs[-2:])
# print('inputs[-1]:', inputs[-1])

# print('input:', inputs[-3:])
