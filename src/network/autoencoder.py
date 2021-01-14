import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from network import encoder, decoder

class AutoEncoder:

    def __new__(self):
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            self.input_shape = (128, 128, 1)

            self.image_input = tf.keras.layers.Input(shape = self.input_shape, name='input_image')

            self.encoder = encoder.Encoder()
            self.decoder = decoder.Decoder(self.input_shape)

            self.enc = self.encoder(self.image_input)
            
            self.dec = self.decoder(self.enc)

            self.model = tf.keras.Model(self.image_input, self.enc)
            self.model = tf.keras.models.load_model('data/model_original')

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(1e-3),
            loss = tfa.losses.TripletSemiHardLoss()
        )

        return self.model