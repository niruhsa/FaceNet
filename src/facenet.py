from network.autoencoder import AutoEncoder
from dataset.dataset import Dataset

class FaceNet:

    def __init__(self):
        self.model = AutoEncoder()
        self.model.summary()
        
        self.dataset = Dataset()
        self.images, self.labels = self.dataset.load_dataset()

        self.train()

    def train(self):
        self.model.fit(
            x = [ self.images, self.labels ],
            y = self.images,
            batch_size=32,
            epochs=8
        )

if __name__ == "__main__": FaceNet()