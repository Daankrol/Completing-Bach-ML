# from read_data import *

# print(get_pairs(voice_number=1, method='cumulative'))
from sklearn.linear_model import LinearRegression
import numpy as np
model = LinearRegression()

x = [[1, 1], [1, 2], [4, 4], [3, 4]]
x_p1 = [1, 1, 4, 3]
x_d1 = [1, 2, 4, 4]

y_1 = [0, 0, 0, 1]
x = [x1p, x2p, ..., x1d, x2d[x_p1 + x_d1], x2, x3]
y = [y1, y2, [1, 0], [1, 0], [0, 1], [0, 1]]


model.fit([np.array(x_p), np.array(x_d)], np.array(y))

print(model.predict([[4, 4]]))
