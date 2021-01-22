import pandas as pd
import numpy as np
from read_data import get_input_output
from methods import WindowMethod, ProbabilityMethods
import matplotlib.pyplot as plt


def generate_dataframe():
    use_features = True
    inputs, outputs, pitch_key = get_input_output(
        voice_number=0, method=WindowMethod.SHIFT, prob_method=ProbabilityMethods.VALUES, window_size=1, use_features=use_features)
    inputs = np.array(inputs[1:])
    outputs = np.array(outputs[:-1])

    pitch = inputs[:, 0]
    if use_features:
        features = inputs[:, 1:-1]
    duration = inputs[:, -1]

    # Create the DF
    d = {'dur': duration}
    df = pd.DataFrame(data=d, dtype=float)
    
    # Add features
    if use_features:
        features = np.transpose(features)
        df['log_pitch'], df['chroma_x'], df['chroma_y'], df['c5_x'], df['c5_y'] = features
    else:
        df['pitch'] = pitch

    # df['one_hot'] = outputs.tolist()
    # Add one-hot encoding as boolean representations
    for idx, pitch in enumerate(pitch_key):
        df['prob_%d' % pitch] = outputs[:, idx]

    return df, pitch_key


def generate_data():
    use_features = True
    inputs, outputs, pitch_key = get_input_output(
        voice_number=0, method=WindowMethod.SHIFT, prob_method=ProbabilityMethods.VALUES, window_size=1, use_features=use_features)

    inputs = np.array(inputs[1:])
    outputs = np.array(outputs[:-1])

    return inputs, outputs


def print_features(df):
    plot_cols = ['log_pitch', 'chroma_x', 'chroma_y', 'c5_x', 'c5_y']
    plot_features = df[plot_cols]
    fig = plot_features.plot(subplots=True)
    fig = fig[0].get_figure()
    fig.savefig('f.png')
