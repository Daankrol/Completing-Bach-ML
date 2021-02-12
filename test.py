# import tensorflow as tf
import numpy as np
from process_data import *
from window_generator import WindowGenerator
import tensorflow as tf

for i in [0,1,2,3]:
    print(min(get_voice(i)))