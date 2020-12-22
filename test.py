# from read_data import *

# print(get_pairs(voice_number=1, method='cumulative'))
from sklearn.linear_model import LinearRegression
import numpy as np
model = LinearRegression()

x = [1, 2, 3, 4, 5, 6]
y = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]


model.fit(np.array(x), np.array(y))

model.predict([7])
