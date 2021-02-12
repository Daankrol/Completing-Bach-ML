from process_data import *
import numpy as np
import matplotlib.pyplot as plt

VOICE = 0

def plot_voice():
    voice_with = get_voice(VOICE, False)
    voice_without = get_voice(VOICE, True)

    dur = len(voice_with)

    time = np.arange(0, dur, 1) 
    time = time * (1/16)

    time_without = np.arange(0, len(voice_without), 1) 
    time_without = time_without * (1/16)

    plt.figure(figsize=(10,4))

    plt.clf()

    ax1 = plt.subplot(212)
    plt.plot(time, voice_with, 'b-')

    ax2 = plt.subplot(211,sharex=ax1)
    plt.plot(time_without, voice_without, 'b-')

    plt.setp(ax2.get_xticklabels(), visible=False)

    ax2.set_title("The soprano voice")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("MIDI note")
    ax2.set_ylabel("MIDI note") 

    plt.savefig('voice.png')
    plt.show()

# plot_voice()

def plot_pitch_hist():
    voice = get_voice(VOICE)

    plt.clf()
    plt.hist(voice, bins=np.unique(np.array(voice)))
    plt.title("MIDI note occurence")
    plt.xlabel("MIDI note")
    plt.ylabel("Frequency")
    plt.savefig('pitch_hist.png')
    plt.show()

# plot_pitch_hist()

def get_all_shifts(voice):
    """
    Get key to represent pitch step differences
    :param voice:
    :return:
    """
    shifts = []
    for i in range(len(voice)-1):
        shift = voice[i+1] - voice[i]
        shifts.append(shift)

    return shifts

def plot_shift_hist():
    voice = get_voice(VOICE)
    shifts = get_all_shifts(voice)

    plt.figure(figsize=(4.5,4))

    plt.clf()
    plt.hist(shifts, bins=np.unique(np.array(shifts)), edgecolor="black")
    plt.title("MIDI shift occurence")
    plt.xlabel("MIDI value with respect to its predecessor")
    plt.ylabel("Frequency")
    plt.ylim(0, 180)

    sub_axes = plt.axes([.62, .6, .25, .25]) 
    sub_axes.hist(shifts, bins=np.unique(np.array(shifts)), edgecolor="black")

    plt.savefig('shift_hist.png')
    plt.show()

# plot_shift_hist()

def plot_predicted_logistic():

    file_input = open("supreme_bach_nice.txt", 'r')
    sounds = file_input.readlines()

    predictions = []
    for i in range(0, len(sounds)):
        predictions.append(int(sounds[i]))

    plt.figure(figsize=(10,4))

    voice = get_voice(VOICE)[-500:]
    plt.clf()
    plt.plot(voice, 'b-')
    plt.plot(range(len(voice), len(voice)+len(predictions)), predictions, 'r-')
    plt.title("A predicted continuation of the soprano voice")
    plt.ylabel("MIDI note value")
    plt.xlabel("index")

    zoom=40

    sub_axes = plt.axes([.22, .65, .2, .2])
    sub_axes.plot(range(len(voice)-len(voice[-zoom:]), len(voice)), voice[-zoom:], 'b-')
    plt.plot(range(len(voice), len(voice)+len(predictions[:zoom])), predictions[:zoom], 'r-')

    plt.savefig('logistic_predicted.png')
    plt.show()

plot_predicted_logistic()