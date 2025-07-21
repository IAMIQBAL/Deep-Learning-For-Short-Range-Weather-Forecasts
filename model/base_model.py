import tensorflow as tf
from tensorflow import keras
from keras import layers

class BaseModel(tf.keras.Model):
    def __init__(self, n_layers, n_filters, kernel_size):
        super(BaseModel, self).__init__()

        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv_lstms = []
        self.batch_norm = []

        for i in range(self.n_layers):
            conv_lstm = layers.ConvLSTM2D(filters = self.n_filters[i],
                                        kernel_size = self.kernel_size[i],
                                        padding = 'same',
                                        return_sequences = True,
                                        activation = 'relu')
            
            if i != self.n_layers - 1:
                bn = layers.BatchNormalization()
                self.batch_norm.append(bn)

            self.conv_lstms.append(conv_lstm)

        self.cnn = layers.Conv3D(filters=1,
                                   kernel_size=(3,3,3),
                                   activation='sigmoid',
                                   padding='same')
    
    def call(self, input):
        x = self.conv_lstms[0](input)
        x = self.batch_norm[0](x)
        for i in range(1, self.n_layers):
            x = self.conv_lstms[i](x)
            if i != self.n_layers - 1:
                x = self.batch_norm[i](x)
        x = self.cnn(x)
        return x