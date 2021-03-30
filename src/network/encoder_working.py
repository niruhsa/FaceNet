import tensorflow as tf
from tensorflow.keras import Model, layers

class Encoder(Model):

    def __init__(self):
        super(Encoder, self).__init__()

    def create_block(self, inputs, filters, kernels_low, kernels_high):
        net = layers.Conv2D(filters, kernels_low, strides = 1, padding = 'same')(inputs)
        net = layers.DepthwiseConv2D(kernels_low, strides = 1, padding = 'same')(net)
        net = layers.BatchNormalization()(net)

        net = layers.Conv2D(filters, kernels_high, strides = 1, padding = 'same')(net)
        net = layers.DepthwiseConv2D(kernels_high, strides = 1, padding = 'same')(net)
        net = layers.BatchNormalization()(net)
        
        net = layers.Conv2D(filters * 2, 1, strides = 2, padding = 'same')(net)
        net = layers.PReLU(shared_axes = [1, 2])(net)

        return net

    def call(self, inputs, training = True):
        inputs = layers.Input(shape = inputs)

        net = self.create_block(inputs, 8, 1, 3)
        net = self.create_block(net, 16, 1, 3)
        net = self.create_block(net, 32, 1, 3)
        net = self.create_block(net, 64, 1, 3)
        net = self.create_block(net, 128, 1, 3)
        net = self.create_block(net, 256, 1, 3)
        net = self.create_block(net, 512, 1, 3)
        
        net = layers.Flatten()(net)

        net = layers.Dense(128)(net)

        net = tf.math.l2_normalize(net, axis = 1, epsilon=1e-3)

        return Model(inputs, net, name = 'facenet')