import json
import numpy as np


class JSONwriter:

    def __init__(self, model, fname):
        self.model = model
        self.fname = fname
        self.json_string = ""
        self.idx = 0

    def save(self):
        self.__model_to_JSON(self.model)
        self.__save_json_string()
    
    
    def __model_to_JSON(self, model):
        # Initialization
        conf = model.get_config()['layers']
        md = {'model_type':'Sequential'}
        md['descriptors'] = []
        
        # Find the descriptors and transform the weights
        weights = []
        w_org = model.get_weights()
        
        # First the input layer
        inp_sizes = conf[0]['config']['batch_input_shape']
        layer_input = {}
        if len(inp_sizes) == 3:
            layer_input = {'layer':'Input2D', 'height':1, 'width':inp_sizes[1], 'channel':inp_sizes[2]}
        elif len(inp_sizes) == 4:
            layer_input = {'layer':'Input2D', 'height':inp_sizes[1], 'width':inp_sizes[2], 'channel':inp_sizes[3]}
        elif len(inp_sizes) == 2:
            layer_input = {'layer':'Input2D', 'height':1, 'width':1, 'channel':inp_sizes[1]}
        if inp_sizes[0] is None:
            layer_input['batch'] = 1
        else:
            layer_input['batch'] = inp_sizes[0]
        md['descriptors'].append(layer_input)
            
        # Remaining layers
        num = len(conf)
        for idx in range(0, num):
            layer = self.__get_layer(conf[idx])
            self.__get_weight(weights, w_org, conf[idx])
            for l in layer:
                md['descriptors'].append(l)
            
        md['weights'] = weights
        
        # Create JSON string
        self.json_string = json.JSONEncoder().encode(md)
    
    def __get_layer(self, layer_descr):
        layers = []
        name = layer_descr['class_name']
        
        if 'BatchNormalization' == name:
            eps = layer_descr['config']['epsilon']
            layers.append({'layer':'BatchNormalization', 'epsilon':eps})
            return layers

        elif 'Conv1D' == name:
            k_s = layer_descr['config']['kernel_size'][0]
            k_num = layer_descr['config']['filters']
            stride = layer_descr['config']['strides'][0]
            layers.append({'layer':'Convolution1D', 'kernel_size':k_s, 'kernel_num':k_num, 'stride':stride, 'padding':0})
            
            if layer_descr['config']['use_bias']:
                layers.append({'layer':'Bias2D', 'units':k_num})
            self.__get_activation(layers, layer_descr)
            return layers
        
        elif 'Conv2D' == name:
            k_h = layer_descr['config']['kernel_size'][0]
            k_w = layer_descr['config']['kernel_size'][1]
            k_num = layer_descr['config']['filters']
            s_h = layer_descr['config']['strides'][1]
            s_v = layer_descr['config']['strides'][0]
            layers.append({'layer':'Convolution2D', 'kernel_height':k_h, 'kernel_width':k_w, 'kernel_num':k_num, 'stride_hz':s_h, 'stride_vl':s_v, 'padding_hz':0, 'padding_vl':0})
            
            if layer_descr['config']['use_bias']:
                layers.append({'layer':'Bias2D', 'units':k_num})
            self.__get_activation(layers, layer_descr)
            return layers 
            
        elif 'Cropping1D' == name:
            trimB = layer_descr['config']['cropping'][0]
            trimE = layer_descr['config']['cropping'][1]
            layers.append({'layer':'Cropping1D', 'trimBegin':trimB, 'trimEnd':trimE})
            return layers
            
        elif 'Cropping2D' == name:
            topTrim = layer_descr['config']['cropping'][0][0]
            bottomTrim = layer_descr['config']['cropping'][0][1]
            leftTrim = layer_descr['config']['cropping'][1][0]
            rightTrim = layer_descr['config']['cropping'][1][1]
            layers.append({'layer':'Cropping2D', 'topTrim':topTrim, 'bottomTrim':bottomTrim, 'leftTrim':leftTrim, 'rightTrim':rightTrim})
            return layers
            
        elif 'Activation' == name:
            self.__get_activation(layers, layer_descr)
            return layers
            
        elif 'Dense' == name:
            layers.append({'layer':'Dense2D', 'units':layer_descr['config']['units']})
            if layer_descr['config']['use_bias']:
                layers.append({'layer':'Bias2D', 'units':layer_descr['config']['units']})
            self.__get_activation(layers, layer_descr)
            return layers

        elif 'Dropout' == name:
            layers.append({'layer':'Dropout', 'rate':layer_descr['config']['rate']})
            return layers
            
        elif 'AveragePooling1D' == name:
            k_s = layer_descr['config']['pool_size'][0]
            stride = layer_descr['config']['strides'][0]
            layers.append({'layer':'AvgPooling1D', 'kernel_size':k_s, 'stride':stride, 'padding':0})
            return layers
            
        elif 'AveragePooling2D' == name:
            k_h = layer_descr['config']['pool_size'][0]
            k_w = layer_descr['config']['pool_size'][1]
            s_h = layer_descr['config']['strides'][1]
            s_v = layer_descr['config']['strides'][0]
            layers.append({'layer':'AvgPooling2D', 'kernel_height':k_h, 'kernel_width':k_w, 'stride_hz':s_h, 'stride_vl':s_v, 'padding_hz':0, 'padding_vl':0})
            return layers 
            
        elif 'MaxPooling1D' == name:
            k_s = layer_descr['config']['pool_size'][0]
            stride = layer_descr['config']['strides'][0]
            layers.append({'layer':'MaxPooling1D', 'kernel_size':k_s, 'stride':stride, 'padding':0})
            return layers
            
        elif 'MaxPooling2D' == name:
            k_h = layer_descr['config']['pool_size'][0]
            k_w = layer_descr['config']['pool_size'][1]
            s_h = layer_descr['config']['strides'][1]
            s_v = layer_descr['config']['strides'][0]
            layers.append({'layer':'MaxPooling2D', 'kernel_height':k_h, 'kernel_width':k_w, 'stride_hz':s_h, 'stride_vl':s_v, 'padding_hz':0, 'padding_vl':0})
            return layers
            
        elif 'GlobalMaxPooling1D' == name:
            layers.append({'layer':'GlobalMaxPooling1D'})
            return layers 
        
        elif 'GlobalMaxPooling2D' == name:
            layers.append({'layer':'GlobalMaxPooling2D'})
            return layers
        
        elif 'GlobalAveragePooling1D' == name:
            layers.append({'layer':'GlobalAveragePooling1D'})
            return layers 
        
        elif 'GlobalAveragePooling2D' == name:
            layers.append({'layer':'GlobalAveragePooling2D'})
            return layers
        
        elif 'Flatten' == name:
            layers.append({'layer':'Flatten'})
            return layers
            
        elif 'Reshape' == name:
            h = layer_descr['config']['target_shape'][0]
            w = layer_descr['config']['target_shape'][1]
            c = layer_descr['config']['target_shape'][2]
            if c is None:
                c = 1
            layers.append({'layer':'Reshape', 'height':h, 'width':w, 'channel':c})
            return layers
        
        elif 'Permute' == name:
            dim1 = layer_descr['config']['dims'][0]
            dim2 = layer_descr['config']['dims'][1]
            dim3 = layer_descr['config']['dims'][2]
            if dim3 is None:
                dim3 = 3
            layers.append({'layer':'Permute', 'dim1':dim1, 'dim2':dim2, 'dim3':dim3})
            return layers
            
        elif 'RepeatVector' == name:
            num = layer_descr['config']['n']
            layers.append({'layer':'RepeatVector', 'num':num})
            return layers

        elif 'SimpleRNN' == name:
            units = layer_descr['config']['units']
            input_dim = self.model.get_weights()[self.idx].shape[0]
            activation = layer_descr['config']['activation']
            layers.append({'layer':'SimpleRNN', 'units':units, 'input_dim':input_dim, 'activation':activation})
            return layers

        elif 'LSTM' == name:
            units = layer_descr['config']['units']
            input_dim = self.model.get_weights()[self.idx].shape[0]
            activation = layer_descr['config']['activation']
            rec_act = layer_descr['config']['recurrent_activation']
            layers.append({'layer':'LSTM', 'units':units, 'input_dim':input_dim, 'activation':activation, 'rec_act':rec_act})
            return layers

        elif 'GRU' == name:
            units = layer_descr['config']['units']
            input_dim = self.model.get_weights()[self.idx].shape[0]
            activation = layer_descr['config']['activation']
            rec_act = layer_descr['config']['recurrent_activation']
            layers.append({'layer':'GRU', 'units':units, 'input_dim':input_dim, 'activation':activation, 'rec_act':rec_act})
            return layers

        elif 'LeakyReLU' == name:
            layers.append({'layer': 'LeakyReLU'})
            return layers

        else:
            raise NotImplementedError("Unknown layer type: " + name)  
    
    def __get_weight(self, weights, w_org, config):
        name = config['class_name']
        
        if 'BatchNormalization' == name:
            w = np.ndarray((1,1, w_org[self.idx].shape[0], 4))
            w[0,0,:,0] = w_org[self.idx][:]
            w[0,0,:,1] = w_org[self.idx + 1][:]
            w[0,0,:,2] = w_org[self.idx + 2][:]
            w[0,0,:,3] = w_org[self.idx + 3][:]
            weights.append(w.tolist())
            self.idx += 4

        elif 'Conv1D' == name:
            w = np.ndarray((1,w_org[self.idx].shape[0], w_org[self.idx].shape[1], w_org[self.idx].shape[2]))
            w[0,:,:, :] = w_org[self.idx][:,:,:]
            weights.append(w.tolist())
            self.idx += 1
            if config['config']['use_bias']:
                w = np.ndarray((1,1,1,w_org[self.idx].shape[0]))
                w[0,0,0, :] = w_org[self.idx][:]
                weights.append(w.tolist())
                self.idx += 1
        
        elif 'Conv2D' == name:
            weights.append(w_org[self.idx].tolist())
            self.idx += 1
            if config['config']['use_bias']:
                w = np.ndarray((1,1,1,w_org[self.idx].shape[0]))
                w[0,0,0, :] = w_org[self.idx][:]
                weights.append(w.tolist())
                self.idx += 1
                
        elif 'Dense' == name:
            w = np.ndarray((1,1,w_org[self.idx].shape[0], w_org[self.idx].shape[1]))
            w[0,0,:, :] = w_org[self.idx][:,:]
            weights.append(w.tolist())
            self.idx += 1
            if config['config']['use_bias']:
                w = np.ndarray((1,1,1,w_org[self.idx].shape[0]))
                w[0,0,0, :] = w_org[self.idx][:]
                weights.append(w.tolist())
                self.idx += 1

        elif 'SimpleRNN' == name:
            w = np.zeros((1, max([w_org[self.idx].shape[0], w_org[self.idx].shape[1]]) , w_org[self.idx].shape[1], 3))
            for v in range(0, w_org[self.idx].shape[0]):
               w[0,v,:,0] = w_org[self.idx][v, :]
            for v in range(0, w_org[self.idx + 1].shape[0]):
               w[0,v,:,1] = w_org[self.idx + 1][v, :]
            w[0,0,:,2] = w_org[self.idx + 2][:]
            weights.append(w.tolist())
            self.idx += 3

        elif 'LSTM' == name:
            in_dim = w_org[self.idx].shape[0]
            units = int(w_org[self.idx].shape[1]/4)
            w = np.zeros((units, max([w_org[self.idx].shape[0], units]), units, 12))
            w[:, 0:in_dim, 0, 0] = np.transpose(w_org[self.idx][:, 0:units]) # W_I
            w[:, 0:in_dim, 0, 1] = np.transpose(w_org[self.idx][:, units:2*units]) # W_F
            w[:, 0:in_dim, 0, 2] = np.transpose(w_org[self.idx][:, 2*units:3*units]) # W_C
            w[:, 0:in_dim, 0, 3] = np.transpose(w_org[self.idx][:, 3*units:4*units]) # W_O
            
            w[:, 0:units, 0, 4] = np.transpose(w_org[self.idx+1][:, 0:units]) # U_I
            w[:, 0:units, 0, 5] = np.transpose(w_org[self.idx+1][:, units:2*units]) # U_F
            w[:, 0:units, 0, 6] = np.transpose(w_org[self.idx+1][:, 2*units:3*units]) # U_C
            w[:, 0:units, 0, 7] = np.transpose(w_org[self.idx+1][:, 3*units:4*units]) # U_O

            w[0, 0, :, 8] = w_org[self.idx+2][0:units] # b_I
            w[0, 0, :, 9] = w_org[self.idx+2][units:2*units] # b_F
            w[0, 0, :, 10] = w_org[self.idx+2][2*units:3*units] # b_C
            w[0, 0, :, 11] = w_org[self.idx+2][3*units:4*units] # b_O
            weights.append(w.tolist())
            self.idx += 3 

        elif 'GRU' == name:
            in_dim = w_org[self.idx].shape[0]
            units = int(w_org[self.idx].shape[1]/3)
            w = np.zeros((units, max([w_org[self.idx].shape[0], units]), units, 9))
            w[:, 0:in_dim, 0, 0] = np.transpose(w_org[self.idx][:, 0:units]) # W_Z
            w[:, 0:in_dim, 0, 1] = np.transpose(w_org[self.idx][:, units:2*units]) # W_R
            w[:, 0:in_dim, 0, 2] = np.transpose(w_org[self.idx][:, 2*units:3*units]) # W_HH
            
            w[:, 0:units, 0, 3] = np.transpose(w_org[self.idx+1][:, 0:units]) # U_Z
            w[:, 0:units, 0, 4] = np.transpose(w_org[self.idx+1][:, units:2*units]) # U_R
            w[:, 0:units, 0, 5] = np.transpose(w_org[self.idx+1][:, 2*units:3*units]) # U_HH

            w[0, 0, :, 6] = w_org[self.idx+2][0:units] # b_Z
            w[0, 0, :, 7] = w_org[self.idx+2][units:2*units] # b_R
            w[0, 0, :, 8] = w_org[self.idx+2][2*units:3*units] # b_HH
            weights.append(w.tolist())
            self.idx += 3            

    def __get_activation(self, layers, layer_dscp):
        activation_name = layer_dscp['config']['activation']
        if activation_name == 'linear':
            pass
        elif activation_name == 'relu':
            layers.append({'layer':'ReLu'})
        elif activation_name == 'softmax':
            layers.append({'layer':'Softmax'})
        elif activation_name == 'elu':
            layers.append({'layer':'ELu'})
        elif activation_name == 'hard_sigmoid':
            layers.append({'layer':'HardSigmoid'})
        elif activation_name == 'sigmoid':
            layers.append({'layer':'Sigmoid'})
        elif activation_name == 'softplus':
            layers.append({'layer':'SoftPlus'})
        elif activation_name == 'softsign':
            layers.append({'layer':'SoftSign'})
        elif activation_name == 'tanh':
            layers.append({'layer':'TanH'})
        else:
            raise NotImplementedError("Unknown Activation type.")      
    
    def __save_json_string(self):
        with open(self.fname, 'w') as f:
            f.write(self.json_string)
