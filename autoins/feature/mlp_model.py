import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tf_layer

class MlpModel(tf.keras.Model):
    def __init__(self, 
                    node_list,
                    activation_list):
        super(MlpModel, self).__init__()    

        self.layer_list = []
        for node, activation in zip(node_list, activation_list):
            self.layer_list.append(tf_layer.Dense(node, activation=activation))
    
    def call(self, inputs):
        outputs = inputs
        for layer in self.layer_list:
            outputs = layer(outputs)
        return outputs