from network.autoencoder import AutoEncoder
import numpy as np
import argparse
import cv2, os
import tensorflow as tf
import random

for gpu in tf.config.experimental.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(gpu, True)

class Test:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.model = AutoEncoder(compile = False)
        self.model.summary()
        self.model.load_weights('data/weights_small.h5')

        self.calculate_distance()

    def load_image(self, image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = image / 255.

        return image

    def get_average_encoding(self, directory):
        preds = []
        for _, _, files in os.walk(directory):
            batch = []
            self.choice = os.path.join(directory, random.choice(files))
            for file in [ random.choice(files) for i in range(128) ]:
                index = files.index(file)
                file = os.path.join(directory, file)
                image = self.load_image(file)
                batch.append(image)

                if len(batch) == 128 or index == (len(files) - 1):
                    preds.extend(self.model(np.array(batch)))
                    batch = []
        return preds

    def get_other_encodings(self, directory):
        completed = []
        for _, subdirs, _ in os.walk(directory):
            random.shuffle(subdirs)
            for subdir in subdirs:
                path = os.path.join(directory, subdir)
                for _, _, files in os.walk(path):
                    preds = self.model(np.array([
                        self.load_image(os.path.join(path, random.choice(files))) for i in range(128)
                    ]))
                    completed.append({ 'person': subdir, 'predictions': preds })

        return completed

    def calculate_distance(self):
        average_encoding = self.get_average_encoding(self.kwargs['person'])
        other_encodings = self.get_other_encodings(self.kwargs['compare'])
        choice = self.model(np.array([ self.load_image(self.kwargs['image']) ]))[0]
        distance = np.min(np.linalg.norm(average_encoding - choice))

        lowest = 100
        for person in other_encodings:
            for prediction in person['predictions']:
                pred_dist = np.min(np.linalg.norm(average_encoding - prediction))
                if pred_dist <= distance:
                    print('[ OK ] Person {} has lower score of {} compared to {}'.format(person['person'], pred_dist, distance))
                    break
                else:
                    if pred_dist < lowest:
                        lowest = pred_dist

        print('[ OK ] Successful with distance {} and second lowest distance of {}'.format(distance, lowest))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--person', type=str, help='Directory of person of face you are comparing')
    args.add_argument('--compare', type=str, help='Directory of dataset of people')
    args.add_argument('--image', type=str, help='Image of face')
    args = args.parse_args()

    Test(**vars(args))

