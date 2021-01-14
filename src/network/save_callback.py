import tensorflow as tf
import time

class SaveCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs = None):
        if epoch % 8 == 0:
            self.model.save('data/model')
            self.model.save_weights('data/model/weights.h5')