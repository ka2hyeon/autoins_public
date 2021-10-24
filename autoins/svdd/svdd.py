import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as tf_optimizers

from autoins.svdd.model import SvddModel, SvddAEModel
from autoins.common import common, io


class SvddDataset():
    def __init__(self, io_manager):
        self.io_manager = io_manager

    def get_dataset(self):
        ag_shape = self.io_manager.ag_demo[0].shape[1:]
        return tf.data.Dataset.from_generator(
                    self._data_generator,
                    output_types = (tf.float32, tf.float32),
                    output_shapes = (ag_shape, ag_shape))

    def _data_generator(self):
        ag_demo = self.io_manager.ag_demo
        ag_concat = np.concatenate(ag_demo,0)
        for i in range(len(ag_concat)):
            yield (ag_concat[i,:], ag_concat[i,:])


class Svdd():
    def __init__(self, 
                    exp_dir,
                    data_name,
                    ccl_id,
                    node_list,
                    l2_reg,
                    ag_shape,
                    quantile,
                    initialize):

        self.exp_dir = exp_dir
        self.data_name = data_name
        self.ccl_id = ccl_id
        self.node_list = node_list
        self.l2_reg = l2_reg
        self.ag_shape = ag_shape
        self.quantile = quantile

        self.io_manager = io.IoManager(exp_dir = self.exp_dir,
                                        data_name =  self.data_name,
                                        ccl_id = self.ccl_id)

        self.svdd_dataset = SvddDataset(self.io_manager)
        self.model =  SvddModel(node_list = self.node_list, l2_reg = self.l2_reg)
        self.ae_model = SvddAEModel(encoder_model = self.model, output_dim = self.ag_shape)
        self.c_thresh = None

    def pretrain(self, nb_pretraining):
        #dataset = self.svdd_dataset.get_dataset()
        #dataset = dataset.batch(32)
        x_data = self.get_data()

        print('pretrain svdd')
        self.ae_model.compile(optimizer="Adam", loss="mse")
        self.ae_model.fit(x_data, x_data, epochs = nb_pretraining)

        save_dir = self.io_manager.svdd_weight_dir
        self.ae_model.save_weights(f'{save_dir}/model.ckpt')

        c_predict = self.model(x_data).numpy()
        self.c0 = np.mean(c_predict, keepdims = True)

    def train(self, nb_training):

        x_data = self.get_data()
        c_data = np.tile(self.c0, [x_data.shape[0], 1])

        print('train svdd')
        optimizer = tf_optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        self.model.fit(x_data, c_data, epochs = nb_training)

        save_dir = self.io_manager.svdd_weight_dir
        self.model.save_weights(f'{save_dir}/model.ckpt')
        np.save(f'{save_dir}/c0.npy',c_data)


    def get_data(self):
        ag_demo = self.io_manager.ag_demo
        ag_concat = np.concatenate(ag_demo, 0)
        return ag_concat

    def predict(self, x):
        c0 = self.c_data[0,:]
        predicted = self.model(x, training = False).numpy()
        c = np.mean(np.square(predicted-c0), axis = 1)
        return c

    def predict_binary(self, x):
        c = self.predict(x)

        if self.c_thresh is None:
            data = self.get_data()
            c_data = self.predict(data)

            if self.quantile >= 1:
                self.c_thresh = np.max(c_data)
            else:
                self.c_thresh = np.quantile(c_data, self.quantile)
        
        new_c = np.copy(c)
        new_c[c>self.c_thresh] = 0
        new_c[c<self.c_thresh] = 1
        return new_c

    def predict_nonbinary(self,x):
        c = self.predict(x)

        if self.c_thresh is None:
            data = self.get_data()
            c_data = self.predict(data)

            if self.quantile >= 1:
                self.c_thresh = np.max(c_data)
            else:
                self.c_thresh = np.quantile(c_data, self.quantile)
        new_c = np.copy(c)
        #return self.c_thresh/new_c
        return np.clip(self.c_thresh/new_c, 0, 1)

    def restore(self):
        save_dir = self.io_manager.svdd_weight_dir        
        self.model.load_weights(f'{save_dir}/model.ckpt')   
        self.c_data = np.load(f'{save_dir}/c0.npy')