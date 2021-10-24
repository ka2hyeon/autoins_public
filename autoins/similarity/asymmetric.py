import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tf_layer

class AsymmetricLayer(tf_layer.Layer):
    def __init__(self, unit):
        super(AsymmetricLayer, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.v = self.add_variable('vectors', 
                                    shape =[self.unit, feature_dim], 
                                    dtype = tf.float32)
        self.A_preserve = self.add_variable('A',
                                    shape = [feature_dim, feature_dim],
                                    dtype = tf.float32,
                                    trainable = True)

    def call(self, inputs, training = None):
        f1 = inputs[0] # [None, i, dim]
        f2 = inputs[1] # [None, i, dim]

        if training:
            A = self._make_A_matrix()
        else:
            A = self.A_preserve
        sim = tf.einsum('ijk,kl->ijl', f1, A) #[None, i, dim]
        sim = tf.matmul( tf.expand_dims(sim,2), tf.expand_dims(f2,3))[:,:,:,0] #[None, i, 1]
        sim = tf.divide(sim, tf.linalg.norm(f1, axis = 2, keepdims = True))
        sim = tf.divide(sim, tf.linalg.norm(f2, axis = 2, keepdims = True))
        return sim

    @tf.function
    def _make_A_matrix(self):
        feature_dim = self.v.shape[-1]
        vvT = tf.matmul(tf.expand_dims(self.v,2), tf.expand_dims(self.v,1)) # [M,dim,dim]
        vTv = tf.matmul(tf.expand_dims(self.v,1), tf.expand_dims(self.v,2)) #[M,1,1]
        Am = tf.eye(feature_dim)-2*tf.divide(vvT, vTv) # [M,dim,dim]
        
        A = tf.eye(feature_dim)
        for i in range(self.unit):
            A = tf.matmul(A, Am[i,:,:])
        
        self.A_preserve.assign(A)
        return A

    def get_A(self):
        return self._make_matrix().numpy()
    

class AsymmetricModel(tf.keras.Model):
    def __init__(self, units):
        super(AsymmetricModel, self).__init__()
        self.asymmetric_layer = AsymmetricLayer(units)

    def call(self, inputs):
        sim = self.asymmetric_layer(inputs)
        return sim
