
import keras
from keras import backend as K
from keras.layers import Dense, Add, Flatten, Lambda, Input, Conv2D, BatchNormalization, Activation
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

earlyStop = keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=1,  verbose=1, patience=100, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint(
    "./checkpoint.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='max')
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')


class DQN:
    def __init__(self, input_shape, output_size, continueTraining, name):
        self.name = name
        model = None

        self.input = image_input = Input(shape=input_shape)
        self.conv1 = image_output = Conv2D(
            filters=32, padding='valid', kernel_size=(8, 8), strides=4)(image_input)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        self.conv2 = image_output = Conv2D(
            filters=64, padding='valid', kernel_size=(4, 4), strides=2)(image_output)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        self.conv3 = image_output = Conv2D(
            filters=128, padding='valid', kernel_size=(3, 3), strides=1)(image_output)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        image_output = Flatten()(image_output)

        output_advantage = Dense(512)(image_output)
        output_advantage = BatchNormalization(
            trainable=True)(output_advantage)
        output_advantage = Activation("elu")(output_advantage)
        output_advantage = Dense(output_size, kernel_initializer=keras.initializers.he_uniform(),
                                 bias_initializer=keras.initializers.he_uniform())(output_advantage)
        final_output_advantage = Lambda(
            lambda x: x - K.mean(x, axis=1, keepdims=True))(output_advantage)

        output_value = Dense(512)(image_output)
        output_value = BatchNormalization(
            trainable=True)(output_value)
        output_value = Activation("elu")(output_value)
        final_output_value = Dense(1, kernel_initializer=keras.initializers.he_uniform(),
                                   bias_initializer=keras.initializers.he_uniform())(output_value)

        final_output = Add()(
            [final_output_advantage, final_output_value])

        model = Model(
            inputs=image_input, outputs=final_output)

        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(
            lr=0.001))

        if continueTraining == True:
            model.load_weights(f'./{self.name}.hdf5')

        self.model = model

    def predict(self, input):
        return self.model.predict(input)

    def train(self, x_train, y_train, sample_weight, epochs):
        return self.model.fit(x_train, y_train, epochs=epochs, sample_weight=sample_weight, callbacks=[tensorboard])

    def save_model(self, name=None):
        self.model.save(f"./{name if name != None else self.name}.hdf5")

    def copy_model(self, cnn2):
        self.model.set_weights(cnn2.get_weights())

    def get_weights(self):
        return self.model.get_weights()

# # Take a look at the model summary
