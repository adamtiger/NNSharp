from keras.models import Sequential
from keras.layers import Cropping1D, Reshape

model = Sequential()
    
model.add(Reshape((3, 2, 3) ,input_shape=(3, 3, 2)))
model.compile(optimizer='rmsprop', loss='mse')

print(model.get_config())