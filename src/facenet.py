from network.autoencoder import AutoEncoder
from dataset.dataset import Dataset
import tensorflow as tf

import wandb, os, argparse
from wandb.keras import WandbCallback
from network.save_callback import SaveCallback
wandb.init(project='facenet')

for gpu in tf.config.experimental.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(gpu, True)

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

class FaceNet:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.model = AutoEncoder()

        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir='data/logs', write_graph=True, write_images=True)
        self.save_callback = SaveCallback()

        self.train(BATCH_SIZE = int(self.kwargs['batch_size']))

    def train(self, BATCH_SIZE = 32):
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1/255.,
            validation_split=0.25,
            rotation_range = 360,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            brightness_range = (0.2, 1.8),
            shear_range = 0.4,
            zoom_range = 0.2,
            channel_shift_range = 100,
            horizontal_flip = True,
            vertical_flip = True,

            )
        self.dataset = self.datagen.flow_from_directory(
            directory = self.kwargs['dataset'],
            target_size = (128, 128),
            class_mode='sparse',
            batch_size = BATCH_SIZE,
            shuffle = True)
        
        try: self.model.fit(self.dataset, epochs = 512, callbacks = [ self.tb_callback, self.save_callback, WandbCallback() ], steps_per_epoch = self.kwargs['steps'])
        except KeyboardInterrupt: pass

        #self.model.save('data/model')
        #self.model.save_weights('data/model/weights.h5')
        #self.model.save(os.path.join(wandb.run.dir, "model.h5"))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='../../Datasets/LFW-Cropped/converted/')
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--steps', type=int, default=512)
    args = args.parse_args()

    FaceNet(**vars(args))