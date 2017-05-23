from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import RMSprop, Adam
import KerasModeltoJSON as js
import numpy as np
import time

model = Sequential()
model.add(Conv2D(32, (8, 8), padding='valid', input_shape=(84, 84, 4), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), padding='valid', strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(6))
      
# rmsprop = RMSprop(lr=alpha, epsilon=0.01, clipvalue=1.0, decay=0.01)
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse')

inp = np.random.rand(1, 84, 84, 4)

start = time.process_time()
model.predict(inp, batch_size = 1)
end = time.process_time()

print (end-start)

wrt = js.JSONwriter(model, "tests/test_cnn_model.json")
wrt.save()