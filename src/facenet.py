from network.autoencoder import AutoEncoder
from dataset.dataset import Dataset
import tensorflow as tf
import tensorflow_datasets as tfds

import wandb, os, argparse
from wandb.keras import WandbCallback
from network.save_callback import SaveCallback
wandb.init(project='facenet')

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

class FaceNet:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.model = AutoEncoder()
        self.model.summary()

        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir='data/logs', write_graph=True, write_images=True)
        self.save_callback = SaveCallback()

        self.train()

    def train(self, BATCH_SIZE = 512):
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255., validation_split=0.25)
        self.dataset = self.datagen.flow_from_directory(
            directory = self.kwargs['dataset'],
            target_size = (128, 128),
            color_mode = 'grayscale',
            class_mode='sparse',
            batch_size = BATCH_SIZE,
            shuffle = True)

        print('training')
        
        try: self.model.fit(self.dataset, epochs = 10240, callbacks = [ self.tb_callback, self.save_callback, WandbCallback() ])
        except KeyboardInterrupt: pass

        self.model.save('data/model')
        self.model.save_weights('data/model/weights.h5')
        self.model.save(os.path.join(wandb.run.dir, "model.h5"))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='../../Datasets/LFW-Cropped/converted/')
    args = args.parse_args()

    FaceNet(**vars(args))