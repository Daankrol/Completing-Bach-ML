
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from data_frame import generate_dataframe
import numpy as np
import read_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 label_columns=None):
        # load all input and teacher values
        df = generate_dataframe()

        n = len(df)
        self.train_df = df[0:int(n*0.7)]  # train split 70% of data
        self.val_df = df[int(n*0.7):int(n*0.9)]  # validation split 20% of data
        self.test_df = df[int(n*0.9):]  # test split 10% of data

        # Normalize the data
        train_mean = self.train_df.iloc[:, :6].mean()
        train_std = self.train_df.iloc[:, :6].std()
        self.train_df.iloc[:, :6] = (
            self.train_df.iloc[:, :6] - train_mean) / train_std
        self.val_df.iloc[:, :6] = (
            self.val_df.iloc[:, :6] - train_mean) / train_std
        self.test_df.iloc[:, :6] = (
            self.test_df.iloc[:, :6] - train_mean) / train_std

        # make_plot(train_std, df.iloc[:, :5], train_mean)

        # Store the raw data.
        # self.train_df = train_df
        # self.val_df = val_df
        # self.test_df = test_df

        labels_all = list(df.columns)
        self.label_columns = labels_all[6:]
        self.input_columns = labels_all[:6]

        # Work out the label column indices.
        if self.label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(self.label_columns)}

        # Work out the inputs column indices.
        if self.input_columns is not None:
            self.input_columns_indices = {
                name: i for i, name in enumerate(self.input_columns)}

        self.column_indices = {name: i for i,
                               name in enumerate(self.train_df.columns)}

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

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

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

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Input column names(s): {self.input_columns}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # label columns is a one-hot encoding vector
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # all features as input
        if self.input_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]]
                    for name in self.input_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):

        data = np.array(data, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

###############################################################################################################
###############################################################################################################


def make_plot(train_std, df, train_mean):
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    fig = ax.get_figure()
    fig.savefig("normalized_violin_features.png")


single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1)
multi_step_window = WindowGenerator(input_width=6, label_width=1, shift=1)

for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

# test = single_step_window.train
print(single_step_window.train)

linear = tf.keras.Sequential(
    [tf.keras.layers.Dense(22, activation="softmax")])
print('Input shape:', multi_step_window.example[0].shape)
print('Output shape:', linear(multi_step_window.example[0]).shape)


MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


history = compile_and_fit(linear, single_step_window)
batch = single_step_window.test.take(1)
print(batch, type(batch), np.shape(batch))

prediction = linear.predict(batch)
print(prediction, prediction.shape, type(prediction))


# prediction = tf.reshape(prediction, [-1])  # flatten the prediction
# print(prediction.shape, type(prediction))
# print(prediction)
# print(f'Max value: {max(prediction)}')
# val_performance = {}
# performance = {}
# val_performance['Linear'] = linear.evaluate(single_step_window.val)
# performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
