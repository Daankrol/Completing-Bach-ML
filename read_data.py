import numpy as np


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


def get_input_output(voice_number=0, method='cumulative', window_size=3):
    pairs = get_pitch_duration_pairs(voice_number=voice_number, method=method)

    pitch_array = [p[0] for p in pairs]
    duration_array = [p[1] for p in pairs]

    if method == 'shift':
        pitch_input, output = construct_windows(data=pitch_array, window_size=window_size)
        duration_input, waste = construct_windows(data=duration_array, window_size=window_size)

    if method == 'cumulative':
        pitch_input = []
        duration_input = []
        output = []
        for i in range(window_size-1, len(pairs)):
            if duration_array[i] == 1:
                continue
            else:
                for j in range(1, duration_array[i]+1):
                    p = pitch_array[i-window_size+1:i+1]
                    d = duration_array[i-window_size+1:i]+[j]
                    o = pitch_array[i]

                    pitch_input.append(p)
                    duration_input.append(d)
                    output.append(o)

    inputs = []
    inputs.append(pitch_input, duration_input)

    return (inputs, output)

def get_pitch_duration_pairs(voice_number=0, method='cumulative'):
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
            if method == 'shift':
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


def construct_windows(data, window_size):
    windows_in = []
    windows_out = []
    for i in range(len(data)):
        if i+window_size + 1 > len(data) - 1:  # out of bounds
            break
        w = data[i:i+window_size]  # +1 for y-value
        o = data[i+window_size]
        windows_in.append(w)
        windows_out.append(o)
    return np.array(windows_in), np.array(windows_out)
