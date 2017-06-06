import numpy as np
import KerasModeltoJSON as js
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Dense, Activation, Flatten, MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D
from keras.layers import Reshape, Permute, RepeatVector, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Cropping1D, Cropping2D, BatchNormalization, SimpleRNN
import json

# json writer
def write(fname, output):
    with open(fname, 'w') as fp:
        json.dump(output, fp)

def generate_test_files():

    # CONVOLUTIONS

    # convolution1D tests
    gen_conv_1D_stride_1() # OK
    gen_conv_1D_stride_2() # OK

    # convolution2D tests
    gen_conv_2D_stride_1_1() # OK
    gen_conv_2D_stride_1_2() # OK
    
    # cropping1D tests
    gen_cropping1D_tests() # OK
    
    # cropping2D tests
    gen_cropping2D_tests() # OK

    # POOLING

    # avgpooling1D_tests
    gen_avgpool_1D_stride_1() # OK
    gen_avgpool_1D_stride_2() # OK

    # avgpooling2D_tests
    gen_avgpool_2D_stride_1_1() # OK
    gen_avgpool_2D_stride_1_2() # OK

    # maxpooling1D tests
    gen_maxpool_1D_stride_1() # OK
    gen_maxpool_1D_stride_2() # OK

    # maxpooling2D tests
    gen_maxpool_2D_stride_1_1() # OK
    gen_maxpool_2D_stride_1_2() # OK
    
    # globalmaxpooling1D tests
    gen_globalmaxpooling1D() # OK
    
    # globalmaxpooling2D tests
    gen_globalmaxpooling2D() # OK
    
    # globalaveragepooling1D tests
    gen_globalaveragepooling1D() # OK
    
    # globalaveragepooling2D tests
    gen_globalaveragepooling2D() # OK
    
    # CORE
    
    # flatten tests
    gen_flatten() # OK

    # dense tests
    gen_dense_units_4() # OK
    
    # reshape tests
    gen_reshape_tests() # OK
    
    # permute tests
    gen_permute_tests() # OK
    
    # repeatvector tests
    gen_repeatvector_tests()

    # ACTIVATIONS

    # ELu test
    gen_elu() # OK

    # HardSigmoid test
    gen_hard_sigmoid() # OK

    # ReLu test
    gen_relu() # OK

    # Sigmoid test
    gen_sigmoid() # OK

    # Softmax test
    gen_softmax() # OK

    # SoftPlus test
    gen_softplus() # OK

    # SoftSign test
    gen_softsign() # OK

    # TanH test
    gen_tanh() # OK

	# NORMALIZATION

	# BatchNormalization
    gen_batchnorm()

    # RNN LAYERS
    # SimpleRNN
    gen_simplernn()

# ---------------------------------------------------------

# CONVOLUTION

def gen_conv_1D_stride_1():
    model = Sequential()
    
    model.add(Conv1D(3, 2, strides=1, input_shape=(6,4)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((2,4,3))
    
    weight[0,0,0] = 0 
    weight[0,0,1] = 1.5
    weight[0,0,2] = 2 
    
    weight[0,1,0] = 0.5
    weight[0,1,1] = -1 
    weight[0,1,2] = -2
    
    weight[0,2,0] = 3 
    weight[0,2,1] = 0
    weight[0,2,2] = 1
     
    weight[0,3,0] = 1
    weight[0,3,1] = -3 
    weight[0,3,2] = 2.5
    
    weight[1,0,0] = 1.5 
    weight[1,0,1] = 0.5
    weight[1,0,2] = -2 
    
    weight[1,1,0] = 1.5
    weight[1,1,1] = -0.5
    weight[1,1,2] = 2.5
    
    weight[1,2,0] = 2.5 
    weight[1,2,1] = 0.5
    weight[1,2,2] = -1.5 
    
    weight[1,3,0] = -1
    weight[1,3,1] = 3 
    weight[1,3,2] = 0.5
    
    bias = np.ndarray(3)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 2.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,6,4))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,0,2] = 2
    inp[0,0,3] = 1.5
     
    inp[0,1,0] = 1
    inp[0,1,1] = 0 
    inp[0,1,2] = 0
    inp[0,1,3] = 0.6
    
    inp[0,2,0] = 2 
    inp[0,2,1] = 1
    inp[0,2,2] = 2 
    inp[0,2,3] = 2.5
    
    inp[0,3,0] = 1
    inp[0,3,1] = 0 
    inp[0,3,2] = -1
    inp[0,3,3] = 0
    
    inp[0,4,0] = 1 
    inp[0,4,1] = -2
    inp[0,4,2] = 3 
    inp[0,4,3] = 3.5
    
    inp[0,5,0] = 2 
    inp[0,5,1] = 1
    inp[0,5,2] = 4 
    inp[0,5,3] = 3.5
    
    wrt = js.JSONwriter(model, "tests/test_conv_1D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_1D_1_output.json", output.tolist())

def gen_conv_1D_stride_2():
    model = Sequential()
    
    model.add(Conv1D(3, 2, strides=2, input_shape=(6,4)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((2,4,3))
    
    weight[0,0,0] = 0 
    weight[0,0,1] = 1.5
    weight[0,0,2] = 2 
    
    weight[0,1,0] = 0.5
    weight[0,1,1] = -1 
    weight[0,1,2] = -2
    
    weight[0,2,0] = 3 
    weight[0,2,1] = 0
    weight[0,2,2] = 1
     
    weight[0,3,0] = 1
    weight[0,3,1] = -3 
    weight[0,3,2] = 2.5
    
    weight[1,0,0] = 1.5 
    weight[1,0,1] = 0.5
    weight[1,0,2] = -2 
    
    weight[1,1,0] = 1.5
    weight[1,1,1] = -0.5
    weight[1,1,2] = 2.5
    
    weight[1,2,0] = 2.5 
    weight[1,2,1] = 0.5
    weight[1,2,2] = -1.5 
    
    weight[1,3,0] = -1
    weight[1,3,1] = 3 
    weight[1,3,2] = 0.5
    
    bias = np.ndarray(3)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 2.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,6,4))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,0,2] = 2
    inp[0,0,3] = 1.5
     
    inp[0,1,0] = 1
    inp[0,1,1] = 0 
    inp[0,1,2] = 0
    inp[0,1,3] = 0.6
    
    inp[0,2,0] = 2 
    inp[0,2,1] = 1
    inp[0,2,2] = 2 
    inp[0,2,3] = 2.5
    
    inp[0,3,0] = 1
    inp[0,3,1] = 0 
    inp[0,3,2] = -1
    inp[0,3,3] = 0
    
    inp[0,4,0] = 1 
    inp[0,4,1] = -2
    inp[0,4,2] = 3 
    inp[0,4,3] = 3.5
    
    inp[0,5,0] = 2 
    inp[0,5,1] = 1
    inp[0,5,2] = 4 
    inp[0,5,3] = 3.5
    
    wrt = js.JSONwriter(model, "tests/test_conv_1D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_1D_2_output.json", output.tolist())

def gen_conv_2D_stride_1_1():
    model = Sequential()
    
    model.add(Conv2D(2, (3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((3,4,2,2))
    
    weight[0,0,0,0] = 0 
    weight[0,0,0,1] = 1.5
    weight[0,0,1,0] = 2 
    weight[0,0,1,1] = 0.5
    
    weight[0,1,0,0] = -1 
    weight[0,1,0,1] = -2
    weight[0,1,1,0] = 3 
    weight[0,1,1,1] = 0
    
    weight[0,2,0,0] = 1 
    weight[0,2,0,1] = 1
    weight[0,2,1,0] = -3 
    weight[0,2,1,1] = 2.5
    
    weight[0,3,0,0] = 1.5 
    weight[0,3,0,1] = 0.5
    weight[0,3,1,0] = -2 
    weight[0,3,1,1] = 1.5
    
    
    weight[1,0,0,0] = -0.5
    weight[1,0,0,1] = 2.5
    weight[1,0,1,0] = 2.5 
    weight[1,0,1,1] = 0.5
    
    weight[1,1,0,0] = -1.5 
    weight[1,1,0,1] = -1
    weight[1,1,1,0] = 3 
    weight[1,1,1,1] = 0.5
    
    weight[1,2,0,0] = 1.5 
    weight[1,2,0,1] = 1
    weight[1,2,1,0] = 0 
    weight[1,2,1,1] = 0
    
    weight[1,3,0,0] = 1.5 
    weight[1,3,0,1] = 0
    weight[1,3,1,0] = -2 
    weight[1,3,1,1] = 3

    
    weight[2,0,0,0] = 1 
    weight[2,0,0,1] = 1
    weight[2,0,1,0] = 2 
    weight[2,0,1,1] = -2
    
    weight[2,1,0,0] = -1 
    weight[2,1,0,1] = 2
    weight[2,1,1,0] = -3 
    weight[2,1,1,1] = 1
    
    weight[2,2,0,0] = 0 
    weight[2,2,0,1] = 1
    weight[2,2,1,0] = -3 
    weight[2,2,1,1] = 2.5
    
    weight[2,3,0,0] = 1.5 
    weight[2,3,0,1] = 2.5
    weight[2,3,1,0] = -2 
    weight[2,3,1,1] = 1.5
    
    bias = np.ndarray(2)
    
    bias[0] = 0.5
    bias[1] = 1.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_conv_2D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_2D_1_output.json", output.tolist())

def gen_conv_2D_stride_1_2():
    model = Sequential()
    
    model.add(Conv2D(2, (2,4), strides=(2,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    weight = np.ndarray((2,4,2,2))
    
    weight[0,0,0,0] = 0 
    weight[0,0,0,1] = 1.5
    weight[0,0,1,0] = 2 
    weight[0,0,1,1] = 0.5
    
    weight[0,1,0,0] = -1 
    weight[0,1,0,1] = -2
    weight[0,1,1,0] = 3 
    weight[0,1,1,1] = 0
    
    weight[0,2,0,0] = 1 
    weight[0,2,0,1] = 1
    weight[0,2,1,0] = -3 
    weight[0,2,1,1] = 2.5
    
    weight[0,3,0,0] = 1.5 
    weight[0,3,0,1] = 0.5
    weight[0,3,1,0] = -2 
    weight[0,3,1,1] = 1.5
    
    
    weight[1,0,0,0] = -0.5
    weight[1,0,0,1] = 2.5
    weight[1,0,1,0] = 2.5 
    weight[1,0,1,1] = 0.5
    
    weight[1,1,0,0] = -1.5 
    weight[1,1,0,1] = -1
    weight[1,1,1,0] = 3 
    weight[1,1,1,1] = 0.5
    
    weight[1,2,0,0] = 1.5 
    weight[1,2,0,1] = 1
    weight[1,2,1,0] = 0 
    weight[1,2,1,1] = 0
    
    weight[1,3,0,0] = 1.5 
    weight[1,3,0,1] = 0
    weight[1,3,1,0] = -2 
    weight[1,3,1,1] = 3
    
    bias = np.ndarray(2)
    
    bias[0] = 0.5
    bias[1] = 1.5
    
    w = [weight, bias]
    model.set_weights(w)
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_conv_2D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_conv_2D_2_output.json", output.tolist())
    
def gen_cropping1D_tests():
    
    model = Sequential()
    
    model.add(Cropping1D(cropping=(1, 2), input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    for l in range(0, 5):
        inp[0, l, 0] = l + 1
        inp[0, l, 1] = -(l + 1)
    
    wrt = js.JSONwriter(model, "tests/test_crop_1D_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_crop_1D_output.json", output.tolist())
    
def gen_cropping2D_tests():
    
    model = Sequential()
    
    model.add(Cropping2D(cropping=((1, 1), (1, 2)), input_shape=(4, 5, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4, 5, 2))
    
    l = 0
    for h in range(0, 4):
        for w in range(0, 5):
            l += 1
            inp[0, h, w, 0] = l + 1
            inp[0, h, w, 1] = -(l + 1)
    
    wrt = js.JSONwriter(model, "tests/test_crop_2D_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_crop_2D_output.json", output.tolist())
    
# POOLING

# avgpooling1D_tests
def gen_avgpool_1D_stride_1():
    model = Sequential()
    
    model.add(AveragePooling1D(pool_size=3, strides=1, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_1D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_1D_1_output.json", output.tolist())
    
def gen_avgpool_1D_stride_2():
    model = Sequential()
    
    model.add(AveragePooling1D(pool_size=3, strides=2, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_1D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_1D_2_output.json", output.tolist())

# avgpooling2D_tests
def gen_avgpool_2D_stride_1_1():
    model = Sequential()
    
    model.add(AveragePooling2D(pool_size=(3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_2D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_2D_1_output.json", output.tolist())
    
def gen_avgpool_2D_stride_1_2():
    model = Sequential()
    
    model.add(AveragePooling2D(pool_size=(3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_avgpool_2D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_avgpool_2D_2_output.json", output.tolist())

# maxpooling1D tests
def gen_maxpool_1D_stride_1():
    model = Sequential()
    
    model.add(MaxPooling1D(pool_size=3, strides=1, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_1D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_1D_1_output.json", output.tolist())
    
def gen_maxpool_1D_stride_2():
    model = Sequential()
    
    model.add(MaxPooling1D(pool_size=3, strides=2, input_shape=(5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,5,2))
    
    inp[0,0,0] = 0 
    inp[0,0,1] = 1
    inp[0,1,0] = 2 
    inp[0,1,1] = 1
    inp[0,2,0] = 0 
    inp[0,2,1] = 0
    inp[0,3,0] = 2 
    inp[0,3,1] = 1
    inp[0,4,0] = 2 
    inp[0,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_1D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_1D_2_output.json", output.tolist())


def gen_maxpool_2D_stride_1_1():
    model = Sequential()
    
    model.add(MaxPooling2D(pool_size=(3,4), strides=(1,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_2D_1_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_2D_1_output.json", output.tolist())
    
def gen_maxpool_2D_stride_1_2():
    model = Sequential()
    
    model.add(MaxPooling2D(pool_size=(2,4), strides=(2,1), input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_maxpool_2D_2_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_maxpool_2D_2_output.json", output.tolist())    

def gen_globalmaxpooling1D():
    
    model = Sequential()
    
    model.add(GlobalMaxPooling1D(input_shape=(3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1, 3, 2))
    
    inp[0, 0, 0] = 1
    inp[0, 1, 0] = 2
    inp[0, 2, 0] = 0
    
    inp[0, 0, 1] = 3
    inp[0, 1, 1] = 4
    inp[0, 2, 1] = 0
    
    wrt = js.JSONwriter(model, "tests/test_globalmaxpool_1D_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_globalmaxpool_1D_output.json", output.tolist())
    
def gen_globalmaxpooling2D():
    
    model = Sequential()
    
    model.add(GlobalMaxPooling2D(input_shape=(3, 3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1, 3, 3, 2))
    
    inp[0, 0, 0, 0] = 1;
    inp[0, 1, 0, 0] = 2;
    inp[0, 2, 0, 0] = 0;

    inp[0, 0, 1, 0] = 3;
    inp[0, 1, 1, 0] = 4;
    inp[0, 2, 1, 0] = 0;

    inp[0, 0, 2, 0] = 2;
    inp[0, 1, 2, 0] = 2;
    inp[0, 2, 2, 0] = 0;


    inp[0, 0, 0, 1] = 0;
    inp[0, 1, 0, 1] = 3;
    inp[0, 2, 0, 1] = 1;

    inp[0, 0, 1, 1] = 1;
    inp[0, 1, 1, 1] = 1;
    inp[0, 2, 1, 1] = -1;

    inp[0, 0, 2, 1] = -3;
    inp[0, 1, 2, 1] = -1;
    inp[0, 2, 2, 1] = 0;
    
    wrt = js.JSONwriter(model, "tests/test_globalmaxpool_2D_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_globalmaxpool_2D_output.json", output.tolist())

def gen_globalaveragepooling1D():
    
    model = Sequential()
    
    model.add(GlobalAveragePooling1D(input_shape=(3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1, 3, 2))
    
    inp[0, 0, 0] = 1
    inp[0, 1, 0] = 2
    inp[0, 2, 0] = 0
    
    inp[0, 0, 1] = 3
    inp[0, 1, 1] = 4
    inp[0, 2, 1] = 0
    
    wrt = js.JSONwriter(model, "tests/test_globalavgpool_1D_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_globalavgpool_1D_output.json", output.tolist())
    
def gen_globalaveragepooling2D():
    
    model = Sequential()
    
    model.add(GlobalAveragePooling2D(input_shape=(3, 3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1, 3, 3, 2))
    
    inp[0, 0, 0, 0] = 1;
    inp[0, 1, 0, 0] = 2;
    inp[0, 2, 0, 0] = 0;

    inp[0, 0, 1, 0] = 3;
    inp[0, 1, 1, 0] = 4;
    inp[0, 2, 1, 0] = 0;

    inp[0, 0, 2, 0] = 2;
    inp[0, 1, 2, 0] = 2;
    inp[0, 2, 2, 0] = 0;


    inp[0, 0, 0, 1] = 0;
    inp[0, 1, 0, 1] = 3;
    inp[0, 2, 0, 1] = 1;

    inp[0, 0, 1, 1] = 1;
    inp[0, 1, 1, 1] = 1;
    inp[0, 2, 1, 1] = -1;

    inp[0, 0, 2, 1] = -3;
    inp[0, 1, 2, 1] = -1;
    inp[0, 2, 2, 1] = 0;
    
    wrt = js.JSONwriter(model, "tests/test_globalavgpool_2D_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_globalavgpool_2D_output.json", output.tolist())

# CORE LAYERS

# FLATTEN
def gen_flatten():
    model = Sequential()
    
    model.add(Flatten(input_shape=(4,5,2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,4,5,2))
    
    inp[0,0,0,0] = 0 
    inp[0,0,0,1] = 1
    inp[0,0,1,0] = 2 
    inp[0,0,1,1] = 1
    inp[0,0,2,0] = 0 
    inp[0,0,2,1] = 0
    inp[0,0,3,0] = 2 
    inp[0,0,3,1] = 1
    inp[0,0,4,0] = 2 
    inp[0,0,4,1] = 1
    
    inp[0,1,0,0] = 0 
    inp[0,1,0,1] = -1
    inp[0,1,1,0] = 1 
    inp[0,1,1,1] = -2
    inp[0,1,2,0] = 3 
    inp[0,1,2,1] = 1
    inp[0,1,3,0] = 2 
    inp[0,1,3,1] = 0
    inp[0,1,4,0] = 2 
    inp[0,1,4,1] = -3
    
    inp[0,2,0,0] = 1 
    inp[0,2,0,1] = 2
    inp[0,2,1,0] = -2 
    inp[0,2,1,1] = 0
    inp[0,2,2,0] = 3 
    inp[0,2,2,1] = -3
    inp[0,2,3,0] = 2 
    inp[0,2,3,1] = 1
    inp[0,2,4,0] = 2 
    inp[0,2,4,1] = 0
    
    inp[0,3,0,0] = 1 
    inp[0,3,0,1] = 2
    inp[0,3,1,0] = 0 
    inp[0,3,1,1] = -2
    inp[0,3,2,0] = 3 
    inp[0,3,2,1] = 1
    inp[0,3,3,0] = 2 
    inp[0,3,3,1] = 3
    inp[0,3,4,0] = -3 
    inp[0,3,4,1] = 1
    
    wrt = js.JSONwriter(model, "tests/test_flat_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_flat_output.json", output.tolist())

# DENSE
def gen_dense_units_4():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_dense_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_dense_output.json", output.tolist())

def gen_reshape_tests():

    model = Sequential()
    
    model.add(Reshape((3, 2, 3), input_shape=(3, 3, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1, 3, 3, 2))
    
    l = 0
    for h in range(0, 3):
        for w in range(0, 3):
            for c in range(0, 2):
                l += 1 
                inp[0, h, w, c] = l + 1
    
    wrt = js.JSONwriter(model, "tests/test_reshape_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_reshape_output.json", output.tolist())
    
def gen_permute_tests():

    model = Sequential()
    
    model.add(Permute((3, 1, 2), input_shape=(2, 3, 4)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1, 2, 3, 4))
    
    l = 0
    for h in range(0, 2):
        for w in range(0, 3):
            for c in range(0, 4):
                l += 1 
                inp[0, h, w, c] = l + 1
    
    wrt = js.JSONwriter(model, "tests/test_permute_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_permute_output.json", output.tolist())      
    
def gen_repeatvector_tests():

    model = Sequential()
    
    model.add(RepeatVector(3, input_shape=(4,)))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((2, 4))

    for l in range(0, 4):
        inp[0, l] = l + 1
        inp[1, l] = -(l + 1)
    
    wrt = js.JSONwriter(model, "tests/test_repeatvector_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_repeatvector_output.json", output.tolist())        
    
# -------------------------------------------------------------------------------------------------------------

# ACTIVATIONS:

# ELu test
def gen_elu():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('elu'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_elu_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_elu_output.json", output.tolist())

# HardSigmoid test
def gen_hard_sigmoid():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('hard_sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_hard_sigmoid_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_hard_sigmoid_output.json", output.tolist())

# ReLu test
def gen_relu():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_relu_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_relu_output.json", output.tolist())

# Sigmoid test
def gen_sigmoid():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_sigmoid_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_sigmoid_output.json", output.tolist())

# Softmax test
def gen_softmax():

    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_softmax_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_softmax_output.json", output.tolist())

# SoftPlus test
def gen_softplus():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('softplus'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_softplus_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_softplus_output.json", output.tolist())

# SoftSign test
def gen_softsign():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('softsign'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_softsign_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_softsign_output.json", output.tolist())

# TanH test
def gen_tanh():
    model = Sequential()
    
    model.add(Flatten(input_shape=(8,1,1)))
    model.add(Dense(4))
    model.add(Activation('tanh'))
    model.compile(optimizer='rmsprop', loss='mse')
    
    inp = np.ndarray((1,8, 1, 1))
    
    inp[0,0,0,0] = 1
    inp[0,1,0,0] = 2
    inp[0,2,0,0] = -1
    inp[0,3,0,0] = 0
    
    inp[0,4,0,0] = 3
    inp[0,5,0,0] = 1
    inp[0,6,0,0] = 1
    inp[0,7,0,0] = 2
    
    weight = np.ndarray((8,4))
    
    weight[0,0] = 0 
    weight[0,1] = 1.5
    weight[0,2] = 2 
    weight[0,3] = 0.5
    
    weight[1,0] = -1 
    weight[1,1] = -2
    weight[1,2] = 3 
    weight[1,3] = 0
    
    weight[2,0] = 1 
    weight[2,1] = 1
    weight[2,2] = -3 
    weight[2,3] = 2.5
    
    weight[3,0] = 1.5 
    weight[3,1] = 0.5
    weight[3,2] = -2 
    weight[3,3] = 1.5
    
    
    weight[4,0] = -0.5
    weight[4,1] = 2.5
    weight[4,2] = 2.5 
    weight[4,3] = 0.5
    
    weight[5,0] = -1.5 
    weight[5,1] = -1
    weight[5,2] = 3 
    weight[5,3] = 0.5
    
    weight[6,0] = 1.5 
    weight[6,1] = 1
    weight[6,2] = 0 
    weight[6,3] = 0
    
    weight[7,0] = 1.5 
    weight[7,1] = 0
    weight[7,2] = -2 
    weight[7,3] = 3
    
    bias = np.ndarray(4)
    
    bias[0] = 0.5
    bias[1] = 1.5
    bias[2] = 1.0
    bias[3] = 3.0
    
    w = [weight, bias]
    model.set_weights(w)
    
    wrt = js.JSONwriter(model, "tests/test_tanh_model.json")
    wrt.save()
    
    output = model.predict(inp, batch_size=1)
    print(output.shape)
    
    write("tests/test_tanh_output.json", output.tolist())

# NORMALIZATION

def gen_batchnorm():
	model = Sequential()
	model.add(BatchNormalization(input_shape=(2, 1, 3)))
	model.compile(optimizer='sgd', loss='mse')

	params = [0] * 4 

	params[0] = np.array([3,3,3]) # gamma
	params[1] = np.array([1,2,-1]) # beta
	params[2] = np.array([2,2,2]) # bias
	params[3] = np.array([5,5,5]) # variance

	data = np.ndarray((4, 2, 1, 3))

	l = 0
	for b in range(0, 4):
		for h in range(0, 2):
			for w in range(0, 1):
				for c in range(0, 3):
					l += 1
					data[b, h, w, c] = l % 7 - 3

	model.set_weights(params)
	output = model.predict(data, batch_size=1)
	
	wrt = js.JSONwriter(model, "tests/test_batchnorm_model.json")
	wrt.save()
    
	print(output.shape)

	write("tests/test_batchnorm_output.json", output.tolist())

# RNN LAYERS
# SimpleRNN
def gen_simplernn():
	model = Sequential()
	model.add(SimpleRNN(4, activation='linear', stateful=False, batch_input_shape=(4, 3, 3)))
	model.compile(optimizer='sgd', loss='mse')

	data = np.ndarray((4, 3, 3))
	kernel = np.ones((3, 4))
	rec_kernel = np.ones((4, 4))
	bias = np.array([1.0, -1.0, 2.0, -4.0])

	k = 0
	for h in range(0, 3):
		for w in range(0, 4):
			k += 1
			kernel[h, w] = k % 5 - 2 

	k = 0
	for h in range(0, 4):
		for w in range(0, 4):
			k += 1
			rec_kernel[h, w] = k % 5 - 2

	parameters = [kernel, rec_kernel, bias]

	model.set_weights(parameters)

	l = 0
	for b in range(0, 4):
		for h in range(0, 3):
			for c in range(0, 3):
				l += 1
				data[b, h, c] = l % 5 + 1

	output = model.predict(data, batch_size=4) # the batch_size has no impact on the result here

	wrt = js.JSONwriter(model, "tests/test_simplernn_model.json")
	wrt.save()
    
	print(output.shape)

	write("tests/test_simplernn_output.json", output.tolist())

# Generate ALL the tests:

generate_test_files()
