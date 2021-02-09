import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from data_frame import generate_dataframe
import numpy as np
from process_data import *
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class WindowGenerator():
    def __init__(self, input_width, label_width, shift, voice_number, batch_size=32,
                 label_columns=None, use_features=True, shuffle=True):
        self.batch_size = batch_size
        # load all input and teacher values
        self.dat, self.shift_conversion_key = extract_features(voice_number=voice_number)

        self.feature_names = ["log_pitch","chroma_x", "chroma_y", "c5_x", "c5_y"]
        self.one_hot_names = []
        for i in range(len(self.shift_conversion_key)):
            self.one_hot_names.append("shift_%d" % i)

        self.column_names = self.feature_names + self.one_hot_names 

        self.df = pd.DataFrame(data = self.dat, dtype=float, columns=self.column_names)
#         # Normalise features
#         self.mean_df = self.df.iloc[:, :5].mean()
#         self.std_df = self.df.iloc[:, :5].std()
#         self.df.iloc[:, :5] = (
#             self.df.iloc[:, :5] - self.mean_df) / self.std_df
        
        labels_all = list(self.df.columns)
        self.label_columns = labels_all[5:]

        # Work out the label column indices.
        if self.label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(self.label_columns)}

        self.column_indices = {name: i for i,
                               name in enumerate(self.df.columns)}

        # split the data
        n = len(self.df)
        self.train_df = self.df[0:int(n*0.7)]  # train split 70% of data
        self.val_df = self.df[int(n*0.7):int(n*0.9)]  # validation split 20% of data
        self.test_df = self.df[int(n*0.9):]  # test split 10% of data
        
        # normalise data
        self.mean_train = self.train_df.iloc[:,:5].mean()
        self.std_train = self.train_df.iloc[:,:5].std()
        
        self.train_df.iloc[:,:5] = (self.train_df.iloc[:,:5] - self.mean_train) / self.std_train
        self.val_df.iloc[:,:5] = (self.val_df.iloc[:,:5] - self.mean_train) / self.std_train
        self.test_df.iloc[:,:5] = (self.test_df.iloc[:,:5] - self.mean_train) / self.std_train
        
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        # self.input_slice = slice(0, self.total_window_size - self.label_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]
        
#         Make timeseries dataset with windows
        self.full_dataset = self.make_dataset(self.df)
        self.train = self.make_dataset(self.train_df, shuffle=shuffle)
        self.val = self.make_dataset(self.val_df, shuffle=shuffle)
        self.test = self.make_dataset(self.test_df, shuffle=shuffle)
        
        
        
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    @property
    def test_no_shuffle(self):
        return self.make_dataset(self.df, shuffle=False).unbatch()    
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Input columns name(s): {self.column_names}'
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
#         print('------------------------')
#         print(inputs)
#         print(labels)
#         print(np.shape(labels))
#         print(self.label_columns)
        # label columns is a one-hot encoding vector
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
#         tf.reshape(labels, [None, self.label_width])
        labels.set_shape([None, self.label_width, None])
        labels = tf.squeeze(labels, axis=1)

        return inputs, labels

    def make_dataset(self, data, shuffle=True):

        data = np.array(data, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)

        return ds

###############################################################################################################
###############################################################################################################


def plot_linear_model_weights(linear_model, window_generator):
    plt.bar(x=range(len(window_generator.train_df.columns)),
            height=linear_model.layers[0].kernel[:, 0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(window_generator.train_df.columns)))
    _ = axis.set_xticklabels(window_generator.train_df.columns, rotation=90)
    plt.show()


def make_plot(train_std, df, train_mean):
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    fig = ax.get_figure()
    fig.savefig("normalized_violin_features.png")


# batch = multiSte.test.take(1)
# print(batch, type(batch), np.shape(batch))

# prediction = linear.predict(batch)
# print(prediction, prediction.shape, type(prediction))


# val_performance = {}
# performance = {}
# val_performance['Linear'] = linear.evaluate(single_step_window.val)
# performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
