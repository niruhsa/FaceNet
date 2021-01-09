import tensorflow as tf
from network import encoder, decoder

class AutoEncoder:

    def __new__(self):
        self.input_shape = (128, 128, 1)

        self.image_input = tf.keras.layers.Input(shape = self.input_shape, name='input_image')
        self.label_input = tf.keras.layers.Input(shape = (1,), name='input_label')

        self.encoder = encoder.Encoder()
        self.decoder = decoder.Decoder(self.input_shape)

        self.enc = self.encoder(self.image_input)
        
        self.dec = self.decoder([ self.enc, self.label_input ])

        self.model = tf.keras.Model(inputs = [ self.image_input, self.label_input ], outputs = self.dec)
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(lr=1e-3),
            loss = 'mse'
        )

        return self.model