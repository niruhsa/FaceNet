import numpy as np
import cv2
import os

class Dataset:

    def load_dataset(self, dataset_dir = '../../Datasets/LFW-Cropped/converted/'):
        self.dataset_dir = dataset_dir
        self.images = {}
        self.people = None
        self.image_dataset = []
        self.label_dataset = []

        # Get all people in directory (names have been converted to integers)
        for _, people, _ in os.walk(os.path.join(dataset_dir)):
            if not self.people:
                self.people = people

        for person in self.people:
            person_dir = os.path.join(self.dataset_dir, person)
            for _, _, faces in os.walk(person_dir):
                for face in faces:
                    image_dir = os.path.join(person_dir, face)
                    image = cv2.imread(image_dir)
                    image = cv2.resize(image, (128, 128))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = image / 255.
                    
                    if person not in list(self.images.keys()):
                        self.images[person] = []
                    
                    self.images[person].append(image)

        for person in list(self.images.keys()):
            images = self.images[person]
            for image in images:
                self.image_dataset.append(image)
                self.label_dataset.append(np.array(person).astype(np.float32))

        print(len(self.image_dataset))
        print(len(self.label_dataset))

        return self.image_dataset, self.label_dataset

    def load_person(self, person):
        person_dir = os.path.join(self.dataset_dir, person)
        for _, _, faces in os.walk(person_dir):
            for face in faces:
                image_dir = os.path.join(person_dir, face)
                image = cv2.imread(image_dir)
                image = cv2.resize(image, (128, 128))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image / 255.
                
                if person not in list(self.images.keys()):
                    self.images[person] = []
                
                self.images[person].append(image)
        return