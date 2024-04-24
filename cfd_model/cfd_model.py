import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, ConvLSTM2D, MaxPool3D, UpSampling3D, LeakyReLU
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.optimizers import Adam


def create_model(input_dimensions):
    GPUS = ["GPU:0","GPU:1"]
    # GPUS = ["GPU:0"]
    strategy = tf.distribute.MirroredStrategy(GPUS)

    with strategy.scope():
        input_dimensions = input_dimensions

        # Encoder
        input = Input(input_dimensions)
        encoder = ConvLSTM2D(filters=64, kernel_size=(4,4), padding="same",
                            return_sequences=True, activation="relu")(input)
        encoder = MaxPool3D(pool_size=(2,2,2))(encoder)
        encoder = ConvLSTM2D(filters=32, kernel_size=(3,3), padding="same",
                            return_sequences=True, activation="relu")(encoder)
        encoder = MaxPool3D(pool_size=(2,2,2))(encoder)
        encoder = ConvLSTM2D(filters=32, kernel_size=(2,2), padding="same",
                            return_sequences=True, activation="relu", name='encoder_output')(encoder)

        # Decoder
        decoder = ConvLSTM2D(filters=32, kernel_size=(2,2), padding="same",
                            return_sequences=True, activation="relu")(encoder)
        decoder = UpSampling3D(size=(2,2,2))(decoder)
        decoder = ConvLSTM2D(filters=32, kernel_size=(3,3), padding="same",
                            return_sequences=True, activation="relu")(decoder)
        decoder = UpSampling3D(size=(2,2,2))(decoder)
        decoder = ConvLSTM2D(filters=64, kernel_size=(4,4), padding="same",
                            return_sequences=True, activation="relu")(decoder)
        decoder = Conv2D(filters=1, kernel_size=(4,4), padding="same",
                            activation=LeakyReLU(), 
                            name='decoder_output')(decoder)

        # Predictor
        predictor = ConvLSTM2D(filters=32, kernel_size=(3,3), padding="same",
                            return_sequences=True, activation="relu")(encoder)
        predictor = UpSampling3D(size=(2,2,2))(predictor)
        predictor = ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same",
                            return_sequences=True, activation="relu")(predictor)
        predictor = UpSampling3D(size=(2,2,2))(predictor)
        predictor = ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same",
                            return_sequences=True, activation="relu")(predictor)
        predictor = Conv2D(filters=1, kernel_size=(3,3), padding="same", 
                            activation=LeakyReLU())(predictor)
        predictor = MaxPool3D(pool_size=(4,1,1), name='predictor_output')(predictor)



        model = Model(inputs=input, outputs=[decoder, predictor])

        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    # model.summary()
    return model

def training(model, X_train, y_train, X_val, y_val):
    batch_size = 16

    # epochs = 2
    epochs = 500

    history = model.fit(
        X_train, [X_train, y_train],
        validation_data=(X_val, [X_val, y_val]),
        batch_size=batch_size, epochs=epochs,
        verbose=1)

    return history, model

def plot_training_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training loss')

    plot_start = 2

    ax1.plot(history.history['decoder_output_loss'][plot_start:])
    ax1.plot(history.history['val_decoder_output_loss'][plot_start:])
    ax1.set_title('Autoencoder loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper right')

    ax2.plot(history.history['predictor_output_loss'][plot_start:])
    ax2.plot(history.history['val_predictor_output_loss'][plot_start:])
    ax2.set_title('Predictor loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'validation'], loc='upper right')

    fig.savefig(model_name + "_training.png")
    fig.show()

def generate_sequence(model, sequence, window):
    dim=sequence.shape
    generated_sequence_size = dim[0]
    generated_sequence = np.zeros(dim)

    # initial condition
    generated_sequence[0:window-1] = sequence[0:window-1]

    for i in range(generated_sequence_size-window+1):
        input = sequence[i:i+window-1]
        predicted_frame = model.predict(np.expand_dims(input, axis=0), verbose=0)[1][0]
        generated_sequence[i+window-1] = predicted_frame
    
    return generated_sequence

def generate_sequences(model, dataset, window):
    generated_dataset = np.zeros(dataset.shape)

    for i, sequence in enumerate(dataset):
        generated_sequence = generate_sequence(model, sequence, window)
        generated_dataset[i,:] = generated_sequence

    # TODO: save file?

    return generated_dataset