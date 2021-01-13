import tensorflow as tf
import numpy as np

class Decoder(tf.keras.Model):

    def __init__(self, input_dim = (128, 128, 1)):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.build_network()

    def build_network(self):
        self.flatten1 = tf.keras.layers.Flatten(name='decoder_flatten')
        self.dense_out = tf.keras.layers.Dense(np.prod(self.input_dim), name='decoder_dense')
        self.reshape = tf.keras.layers.Reshape(self.input_dim, name='decoder_reshape')
        self.l2norm = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    def call(self, inputs, training = True):
        net = self.flatten1(inputs)
        net = self.dense_out(net)
        net = self.reshape(net)
        net = self.l2norm(net)

        return net