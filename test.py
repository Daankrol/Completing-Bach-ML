# import tensorflow as tf
import numpy as np
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

inputs = np.array([[1, 2, 3], [4, 5, 6]])
outputs = np.array([[1, 0, 0], [0, 0, 1]])


inputs = [[1, 2, 3], [4, 5, 6]]
outputs = [[1, 0, 0], [0, 0, 1]]

together = inputs[0] + [outputs[0]]
print(together)

np_t = np.array(together, dtype='object')
print(np_t)

ni = np.array([1, 2, 3])
no = np.array([0, 0, 0, 1])

n1 = np.append(ni, np.array(no))
print(n1)


x = [[1, 2], [1, 2, 3], [1]]
y = np.array([np.array(xi) for xi in x])
print(y)
