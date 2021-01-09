import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()
        self.build_network()

    def build_network(self):
        # 128 x 128 x 1 -> 64 x 64 x 128
        self.conv2d_1_1 = tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same', name='conv2d_1_1')
        self.mp2d_1_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='mp2d_1_1')

        # 64 x 64 x 128 -> 32 x 32 x 128
        self.conv2d_1_2 = tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same', name='conv2d_1_2')
        self.mp2d_1_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='mp2d_1_2')

        # 32 x 32 x 64 -> 16 x 16 x 64
        self.conv2d_2_1 = tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same', name='conv2d_2_1')
        self.mp2d_2_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='mp2d_2_1')

        # 16 x 16 x 64 -> 8 x 8 x 64
        self.conv2d_2_2 = tf.keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same', name='conv2d_2_2')
        self.mp2d_2_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='mp2d_2_2')

        # 8 x 8 x 32 -> 4 x 4 x 32
        self.conv2d_3_1 = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same', name='conv2d_3_1')
        self.mp2d_3_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='mp2d_3_1')

        # 4 x 4 x 32 -> 2 x 2 x 32
        self.conv2d_3_2 = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same', name='conv2d_3_2')
        self.mp2d_3_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='mp2d_3_2')

        self.ft_1 = tf.keras.layers.Flatten(name='ft_1')
        self.dp_1 = tf.keras.layers.Dropout(0.2, name='dp_1')
        self.d_1 = tf.keras.layers.Dense(128, name='d_1')

    def call(self, inputs, training = True):
        net = self.conv2d_1_1(inputs)
        net = self.mp2d_1_1(net)

        net = self.conv2d_1_2(net)
        net = self.mp2d_1_2(net)

        net = self.conv2d_2_1(net)
        net = self.mp2d_2_1(net)

        net = self.conv2d_2_2(net)
        net = self.mp2d_2_2(net)

        net = self.conv2d_3_1(net)
        net = self.mp2d_3_1(net)

        net = self.conv2d_3_2(net)
        net = self.mp2d_3_2(net)

        net = self.ft_1(net)
        net = self.dp_1(net)
        net = self.d_1(net)

        return net