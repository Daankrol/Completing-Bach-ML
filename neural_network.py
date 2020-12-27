import tensorflow as tf
from read_data import get_input_output

# A layers.Dense with no activation set is a linear model.
# The layer only transforms the last axis of the data from
# (batch, time, inputs) to (batch, time, units),
# it is applied independently to every item across the batch and time axes.


linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

MAX_EPOCHS = 20


def compile_and_fit(model, window_train, window_val, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(window_train, epochs=MAX_EPOCHS,
                        validation_data=window_val,
                        callbacks=[early_stopping])
    return history


inputs, outputs, key = get_input_output(
    voice_number=0, method='cumulative', prob_method='values', window_size=16, use_features=True)

history = compile_and_fit(linear, inputs, outputs)

print(history)
