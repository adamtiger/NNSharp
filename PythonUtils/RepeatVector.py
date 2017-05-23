import numpy as np
from keras.models import Sequential
from keras.layers import RepeatVector
from keras import optimizers

inp = 5

model = Sequential()
model.add(RepeatVector(3, input_shape = (inp,)))
model.compile(loss='mse', optimizer='sgd')

input_data = np.ndarray((1, inp))

for i in range(0, inp):
   input_data[0, i] = i * 2 +1

output_data = model.predict(input_data)

print(output_data)
print(output_data.shape)