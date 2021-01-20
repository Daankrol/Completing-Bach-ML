from data_frame import generate_data
import numpy as np


class WindowGenerator2():
    def __init__(self):

        inputs, outputs = generate_data()  # np arrays
        print(inputs[:2])
        print(outputs[:2])
        # model.fit(x=windows_feautres, y=one_hot_encodings)

    def make_dataset(self, data):

        data = np.array(data, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        # ds = ds.map(self.split_window)

        return ds


wg = WindowGenerator2()
