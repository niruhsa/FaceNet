from network.autoencoder import AutoEncoder
from dataset.dataset import Dataset
import tensorflow as tf

class FaceNet:

    def __init__(self):
        self.model = AutoEncoder()
        self.model.summary()
        
        self.dataset = Dataset()
        self.images, self.labels = self.dataset.load_dataset()
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.)
        self.dataset = self.datagen.flow_from_directory(
            directory = '../../Datasets/LFW-Cropped/converted/',
            target_size = (128, 128),
            color_mode = 'grayscale',
            class_mode='binary',
            batch_size = 32,
            shuffle = True)

        self.train()

    def train(self, BATCH_SIZE = 32):
        print('training')
        #data = tf.data.Dataset.zip((self.images, self.labels)).batch(32)
        #print(data)
        self.model.fit(self.dataset, epochs = 8)

if __name__ == "__main__": FaceNet()