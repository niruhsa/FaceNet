import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from network import encoder, decoder

from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
import tensorflow.keras.backend as K

class AutoEncoder:

    def __new__(self, compile = True):
        self.input_shape = (128, 128, 3)

        self.encoder = encoder.Encoder()
        self.enc = self.encoder(self.input_shape)
        self.enc.summary()
        print('Total number of FLOPs: {:,}'.format(get_flops(self.enc)))

        if compile:
            self.enc.compile(
                optimizer = tf.keras.optimizers.Adam(lr = 1e-3),
                loss = tfa.losses.TripletSemiHardLoss(margin = 0.2)
            )

        return self.enc

def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops