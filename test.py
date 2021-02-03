
from process_data import *
import numpy as np
import matplotlib.pyplot as plt


# def get_break_points()

voice0 = get_voice(0, False)

dur = len(voice0)

time = np.arange(0, dur, 1) 
time = time * (1/16)


plt.clf()
plt.plot(time, voice0, 'b-')
plt.title("The soprano voice")
plt.xlabel("Time (s)")
plt.ylabel("MIDI note")
plt.savefig('voices.png')
plt.show()


# from window_generator import WindowGenerator

# # TODO label_columns does not exist yet
# w2 = WindowGenerator(input_width=6, label_width=1, shift=1)
# print(w2)

# example_window = tf.stack([np.array(w2.train_df[:w2.total_window_size]),
#                            np.array(w2.train_df[100:100+w2.total_window_size]),
#                            np.array(w2.train_df[200:200+w2.total_window_size])])

# example_inputs, example_labels = w2.split_window(example_window)

# # print(example_labels)

# # print(example_inputs)

# # print('All shapes are: (batch, time, features)')
# # print(f'Window shape: {example_window.shape}')
# # print(f'Inputs shape: {example_inputs.shape}')
# # print(f'labels shape: {example_labels.shape}')

# # w2.make_dataset()

# inputs = np.array([[1, 2, 3], [4, 5, 6]])
# outputs = np.array([[1, 0, 0], [0, 0, 1]])


# inputs = [[1, 2, 3], [4, 5, 6]]
# outputs = [[1, 0, 0], [0, 0, 1]]

# together = inputs[0] + [outputs[0]]
# print(together)

# np_t = np.array(together, dtype='object')
# print(np_t)

# ni = np.array([1, 2, 3])
# no = np.array([0, 0, 0, 1])

# n1 = np.append(ni, np.array(no))
# print(n1)


# x = [[1, 2], [1, 2, 3], [1]]
# y = np.array([np.array(xi) for xi in x])
# print(y)

# test = [35,42,32]
# voice=test

# shift_key = []
# for i in range(len(voice)-1):
#     shift = voice[i+1] - voice[i]
#     shift_key.append(shift)

# shift_key = list(set(shift_key))

# print(shift_key)

# processed_data = []

# for i in range(1,len(voice)):
#     current, previous = voice[i], voice[i-1]
#     pitch_features = get_pitch_features(current)
#     shift_one_hot = [0] * len(shift_key)
#     pitch_shift = current - previous
#     shift_one_hot[shift_key.index(pitch_shift)] = 1 #set one hod encoing vector for the pitch shift

#     processed_data.append(pitch_features + shift_one_hot)

# print(processed_data)

# data, shift_key = extract_features()

# print("key", len(shift_key), shift_key)

# inputs,outputs = create_inputs_outputs_from_data(data, shift_key, 2)

# last_input = inputs[-1]
# print("last:", last_input)

# probs = [0]*len(shift_key)
# probs[0]=1 # +1
# probs=np.array(probs)
# print("probs:", probs)

# predicted_shift = get_shift_from_probability(probs, shift_key)

# print("pred_shift:", predicted_shift)

# add_predicted_value(inputs, 68, predicted_shift, shift_key)

# print("new last:", inputs[-1])