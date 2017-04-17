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
        conf = model.get_config()
        md = {'model_type':'Sequential'}
        md['descriptors'] = []
        
        # Find the descriptors and transform the weights
        weights = []
        w_org = model.get_weights()
        
        # First the input layer
        inp_sizes = conf[0]['config']['batch_input_shape']
        layer_input = {'layer':'Input2D', 'height':inp_sizes[1], 'width':inp_sizes[2], 'channel':inp_sizes[3]}
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
        
        if 'Conv2D' == name:
            k_h = layer_descr['config']['kernel_size'][0]
            k_w = layer_descr['config']['kernel_size'][1]
            k_num = layer_descr['config']['filters']
            s_h = layer_descr['config']['strides'][1]
            s_v = layer_descr['config']['strides'][0]
            layers.append({'layer':'Convolution2D', 'kernel_height':k_h, 'kernel_width':k_w, 'kernel_num':k_num, 'stride_hz':s_h, 'stride_vl':s_v, 'padding_hz':0, 'padding_vl':0})
            
            if layer_descr['config']['use_bias']:
                layers.append({'layer':'Bias2D', 'units':k_num})
            return layers 
            
        elif 'Activation' == name:
            if layer_descr['config']['activation'] == 'relu':
                layers.append({'layer':'ReLu'})
            elif layer_descr['config']['activation'] == 'softmax':
                layers.append({'layer':'Softmax'})
            else:
                raise NotImplementedError("Unknown Activation type.")
            return layers
            
        elif 'Dense' == name:
            layers.append({'layer':'Dense2D', 'units':layer_descr['config']['units']})
            if layer_descr['config']['use_bias']:
                layers.append({'layer':'Bias2D', 'units':layer_descr['config']['units']})
            return layers
             
        elif 'MaxPooling2D' == name:
            k_h = layer_descr['config']['pool_size'][0]
            k_w = layer_descr['config']['pool_size'][1]
            s_h = layer_descr['config']['strides'][1]
            s_v = layer_descr['config']['strides'][0]
            layers.append({'layer':'MaxPooling2D', 'kernel_height':k_h, 'kernel_width':k_w, 'stride_hz':s_h, 'stride_vl':s_v, 'padding_hz':0, 'padding_vl':0})
            
            return layers 
        elif 'Flatten' == name:
            layers.append({'layer':'Flatten'})
            
            return layers
        else:
            raise NotImplementedError("Unknown layer type: " + name)  
    
    def __get_weight(self, weights, w_org, config):
        name = config['class_name']
        
        if 'Conv2D' == name:
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
            
             
    
    def __save_json_string(self):
        with open(self.fname, 'w') as f:
            f.write(self.json_string)