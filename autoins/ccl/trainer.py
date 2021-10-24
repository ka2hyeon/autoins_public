import numpy as np
import tensorflow as tf
from autoins.common import common, io

class CclTrainer():
    def __init__(self, model, data_generator, labeler, **kwargs):
        self.model = model
        self.data_generator = data_generator
        self.labeler = labeler
        self.exp_dir = kwargs.get('exp_dir')
        self.data_name = kwargs.get('data_name')
        self.ccl_id = kwargs.get('ccl_id')
        self.relabel_freq = kwargs.get('relabel_freq')
        self.batch_size = kwargs.get('batch_size')
        self.nb_epoch = kwargs.get('nb_epoch')
        self.initialize = kwargs.get('initialize')
        self.io_manager = io.IoManager(exp_dir = self.exp_dir,
                                        data_name =  self.data_name,
                                        ccl_id =  self.ccl_id)
        self._count = 0
        if self.initialize:
            self.tensorlog_dir = f'{self.exp_dir}/log/ccl/{self.data_name}/{self.ccl_id}'
            common.create_dir(self.tensorlog_dir, clear_dir = True)

    def fit(self):
        ag_shape = self.model.ag_shape
        N = self.model.contrastive_sample_size
        nb_subgoal = self.model.nb_subgoal

        ## Set dataset
        input_types = dict(
            positive_pair = tf.float32,
            negative_pair = tf.float32,
            positive_dist = tf.float32,
            negative_dist = tf.float32,
            x_classification = tf.float32,
            label = tf.float32)

        input_shapes = dict(
            positive_pair = (1, 2, ag_shape),
            negative_pair = (2*N-1, 2, ag_shape),
            positive_dist = (1,),
            negative_dist = (2*N-1,),
            x_classification = (ag_shape,),
            label = (nb_subgoal))

        self.dataset = tf.data.Dataset.from_generator(
            self._data_generator,
            output_types = (input_types, tf.float32),
            output_shapes = (input_shapes, (1,)) )

        #self.dataset = self.dataset.take(100)
        self.dataset = self.dataset.batch(32)

        ## Set callback
        self.tb_callback = tf.keras.callbacks.TensorBoard(self.tensorlog_dir, update_freq = 1)
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = f'{self.io_manager.ccl_weight_dir}/checkpoint',
            save_weights_only = True)

        ## Fit
        self.model.model.fit(self.dataset, 
                            epochs = self.nb_epoch, 
                            callbacks = [self.model_checkpoint_callback, self.tb_callback])

    def _data_generator(self):
        if self._count % self.relabel_freq == 0:
            adj_mat, label_demo = self.labeler.update_label()
            self.io_manager.adj_mat = adj_mat
            self.io_manager.label = label_demo

        nb_data_gen = self.data_generator.nb_data_gen
        inputs_train = self.data_generator.generate_data()

        for i in range(nb_data_gen):
            batch_dict = dict()
            for key, value in inputs_train.items():
                batch_dict[key] = value[i,:]

            mockup_output = np.asarray([np.nan])
            yield (batch_dict, mockup_output)
