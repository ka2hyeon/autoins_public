import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tf_layers
import tensorflow.keras.models as tf_models
import tensorflow.keras.optimizers as tf_optimizers
import tensorflow.keras.losses as tf_losss
import tensorflow.keras.regularizers as tf_regs

class SvddModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(SvddModel, self).__init__(*args)
        self.node_list = kwargs.get('node_list')
        self.l2_reg = kwargs.get('l2_reg')

        self.layer_list = []
        for i, node in enumerate(self.node_list[:-1]):
            dense_layer = tf_layers.Dense(node, 
                                        activation=None,
                                        use_bias = False,
                                        kernel_regularizer=tf_regs.l2(self.l2_reg))
            activation_layer = tf_layers.ReLU()
            batchnorm_layer = tf_layers.BatchNormalization()
            
            self.layer_list.append(dense_layer)
            self.layer_list.append(activation_layer)
            self.layer_list.append(batchnorm_layer)

        final_dense_layer = tf_layers.Dense(self.node_list[-1], 
                                        activation=None,
                                        use_bias = False,
                                        kernel_regularizer=tf_regs.l2(self.l2_reg))
        self.layer_list.append(final_dense_layer)


    
    def call(self, inputs):
        outputs = inputs
        for layer in self.layer_list:
            outputs = layer(outputs)
        return outputs


class SvddAEModel(tf.keras.Model):
    def __init__(self, encoder_model, output_dim, *args):
        super(SvddAEModel, self).__init__(*args)
        self.output_dim = output_dim
        self.encoder_model = encoder_model
        self.node_list = list(reversed(self.encoder_model.node_list))
        self.l2_reg = self.encoder_model.l2_reg

        self.layer_list = []
        for i, node in enumerate(self.node_list[:-1]):
            dense_layer = tf_layers.Dense(node, 
                                        activation=tf.nn.relu,
                                        use_bias = False,
                                        kernel_regularizer=tf_regs.l2(self.l2_reg))
            
            self.layer_list.append(dense_layer)

        final_dense_layer = tf_layers.Dense(self.output_dim, 
                                        activation=None,
                                        use_bias = False,
                                        kernel_regularizer=tf_regs.l2(self.l2_reg))
        self.layer_list.append(final_dense_layer)
    
    def call(self, inputs):
        outputs = self.encoder_model(inputs)
        for layer in self.layer_list:
            outputs = layer(outputs)
        return outputs
