import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

class DerivedModel(keras.models.Model):
    def __init__(self, n_layers, n_filters, kernel_size):
        super(DerivedModel, self).__init__()

        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv_lstms = []
        self.batch_norms = []
        self.activations = []
        self.drop_out = []

        for i in range(self.n_layers):
            conv_lstm = layers.ConvLSTM2D(filters = self.n_filters[i],
                                        kernel_size = self.kernel_size[i],
                                        padding = 'same',
                                        return_sequences = True,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.01))
            bn = layers.BatchNormalization()
            self.batch_norms.append(bn)

            self.conv_lstms.append(conv_lstm)

            act = layers.Activation('relu')
            self.activations.append(act)

            dropout = layers.Dropout(0.2)
            self.drop_out.append(dropout)

            self.cnn = layers.Conv3D(filters=1,
                                   kernel_size=(3,3,3),
                                   activation='sigmoid',
                                   padding='same')
    
    def call(self, input):
        x = self.conv_lstms[0](input)
        x = self.batch_norms[0](x)
        x = self.activations[0](x)
        x = self.drop_out[0](x)
        for i in range(1, self.n_layers):
            x = self.conv_lstms[i](x)
            x = self.batch_norms[i](x)
            x = self.activations[i](x)
            x = self.drop_out[i](x)
        x = self.cnn(x)
        return x