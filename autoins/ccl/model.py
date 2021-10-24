import time
import traceback
import IPython
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tf_layer
import tensorflow.keras.models as tf_model
import tensorflow.keras.optimizers as tf_optimizer
import tensorflow.keras.losses as tf_loss
import tensorflow.keras.regularizers as tf_reg

from autoins.common import common, io

class CclModel():
    def __init__(self,
                    exp_dir,
                    data_name,
                    ccl_id,
                    ag_shape,
                    nb_subgoal,
                    feature_dim,
                    beta,
                    lamda,
                    c1,
                    learning_rate,
                    contrastive_sample_size,
                    feature_model,
                    similarity_model,
                    classifier_model,
                    initialize = False):
        
        self.exp_dir = exp_dir
        self.data_name = data_name
        self.ccl_id = ccl_id
        self.ag_shape = ag_shape
        self.nb_subgoal = nb_subgoal
        self.feature_dim = feature_dim
        self.beta = beta
        self.lamda = lamda
        self.c1 = c1
        self.learning_rate = learning_rate
        self.contrastive_sample_size = contrastive_sample_size
        self.feature_model = feature_model
        self.similarity_model = similarity_model
        self.classifier_model = classifier_model

        self.model = self.build()
        self.compile(self.model)
        self.io_manager = io.IoManager(exp_dir = self.exp_dir,
                                        data_name =  self.data_name,
                                        ccl_id =  self.ccl_id)


    def build(self):
        inputs_class, loss_class = self._build_classification()
        inputs_contra, loss_contra = self._build_contrastive_learning()

        input_dict = dict()
        input_dict.update(inputs_class)
        input_dict.update(inputs_contra)
        
        total_loss = loss_contra + self.c1*loss_class
        model = tf_model.Model(inputs = input_dict, outputs = total_loss)
        return model

    def compile(self, model):
        loss_fn = lambda y_true, y_pred: y_pred
        optimizer = tf_optimizer.Adam(learning_rate= self.learning_rate) 
        model.compile(optimizer = optimizer, loss = loss_fn)
        return model
    
    def _build_classification(self):
        N = self.contrastive_sample_size
        nb_subgoal = self.nb_subgoal
        ag_shape = self.ag_shape
        feature_dim = self.feature_dim

        x = tf_layer.Input(shape = (ag_shape,), dtype = tf.float32)
        label = tf_layer.Input(shape = (nb_subgoal), dtype = tf.int32)

        feature = self.feature_model(x)
        label_predict = self.classifier_model(feature)
        classification_loss = self._build_classification_loss(label_predict, label)

        inputs = dict(x_classification = x, label = label)
        return inputs, classification_loss

    def _build_contrastive_learning(self):
        N = self.contrastive_sample_size
        nb_subgoal = self.nb_subgoal
        ag_shape = self.ag_shape
        feature_dim = self.feature_dim

        positive_pair = tf_layer.Input(shape = [1, 2, ag_shape], dtype = tf.float32)
        negative_pair = tf_layer.Input(shape = [2*N-1, 2, ag_shape], dtype = tf.float32)
        positive_dist = tf_layer.Input(shape = [1,], dtype = tf.float32)
        negative_dist = tf_layer.Input(shape = [2*N-1,], dtype = tf.float32)    

        x0_positive = positive_pair[:,:,0] 
        x1_positive = positive_pair[:,:,1] 
        x0_negative = negative_pair[:,:,0]
        x1_negative = negative_pair[:,:,1]

        f0_positive = self._build_feature(x0_positive)
        f1_positive = self._build_feature(x1_positive)
        f0_negative = self._build_feature(x0_negative)
        f1_negative = self._build_feature(x1_negative)

        assert f0_positive.shape.as_list() == [None, 1, feature_dim]
        assert f1_positive.shape.as_list() == [None, 1, feature_dim]
        assert f0_negative.shape.as_list() == [None, 2*N-1, feature_dim]
        assert f1_negative.shape.as_list() == [None, 2*N-1, feature_dim]

        gamma_positive = tf.exp(-tf.expand_dims(positive_dist,-1)/self.beta) 
        gamma_negative = tf.exp(-tf.expand_dims(negative_dist,-1)/self.beta) 
        sim_positive = self.similarity_model( [f0_positive, f1_positive] ) 
        sim_negative = self.similarity_model( [f0_negative, f1_negative] )  

        assert sim_positive.shape.as_list() == [None, 1, 1]
        assert sim_negative.shape.as_list() == [None, 2*N-1, 1]

        contrastive_loss = self._build_contrastive_loss(
                                            sim_positive, 
                                            sim_negative, 
                                            gamma_positive, 
                                            gamma_negative, 
                                            lamda = self.lamda)

        inputs = dict(positive_pair = positive_pair,
                        positive_dist = positive_dist,
                        negative_pair = negative_pair,
                        negative_dist = negative_dist)
        return inputs, contrastive_loss

    def _build_feature(self, x):
        '''
        x: [None, i, j, k]
            - [None]: the size of a training batch
            - [i]: the size of pairs  for contrastive learning
            - [j,k]: feature dimension

        f: [None, i]
            - [i]: output feature dimension
        '''

        batch_size = x.shape[1]
        x_shape = list(x.shape)[2:] 

        x_flatten = tf.reshape(x, [-1,]+x_shape)
        feature_flatten = self.feature_model(x_flatten) 
        feature = tf.reshape(feature_flatten, [-1, batch_size, self.feature_dim])
        return feature

    def _build_classification_loss(self, label_predict, label):
        cce = tf_loss.CategoricalCrossentropy()
        loss = cce(label, label_predict)
        return loss

    def _build_contrastive_loss(self, 
                        sim_posi, 
                        sim_neg, 
                        gamma_positive, 
                        gamma_negative, 
                        lamda = 1):
        '''
        sim_positive: [None, 1, 1]
        sim_negative: [None, N, 1]
        '''
        assert sim_posi.shape[1] == 1
        assert len(sim_posi.shape) == len(sim_neg.shape) == 3
        
        epsilon = 1e-30
        positive = lamda*sim_posi[:,:,0] + tf.math.log( tf.maximum(gamma_positive[:,:,0], epsilon))
        negative = lamda*sim_neg[:,:,0]  + tf.math.log(tf.maximum(1-gamma_negative[:,:,0], epsilon))

        concat = tf.concat([positive, negative], 1)
        softmax = tf.nn.softmax(concat, 1)
        p_positive = softmax[:,:1]
        p_negative = softmax[:,1:]

        loss = -tf.math.log( tf.maximum(p_positive, epsilon))
        loss = tf.reduce_mean(loss)
        return loss

    def restore(self):
        path = f'{self.io_manager.ccl_weight_dir}/checkpoint'
        self.model.load_weights(path)


    def predict_similarity(self, x0, x1, input_type = 'feature'):
        """
        f0 : [N, dim]
        f1 : [N, dim]
        """
        if input_type == 'feature':
            f0 = x0
            f1 = x1
        elif input_type == 'state':
            f0 = self.feature_model(x0).numpy()
            f1 = self.feature_model(x1).numpy()
        
        f0_ext = np.expand_dims(f0,1) 
        f1_ext = np.expand_dims(f1,1)
        
        sim = self.similarity_model([f0_ext, f1_ext]).numpy()[:,0,0]
        return sim