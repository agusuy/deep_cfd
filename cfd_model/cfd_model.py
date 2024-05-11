import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Conv2D, ConvLSTM2D, MaxPool3D, UpSampling3D, LeakyReLU
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.optimizers import Adam
from datetime import datetime
from .cfd_dataset import get_dataset
from .model_constants import WINDOW

def get_model(model_file):
    model = load_model(model_file)
    return model

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

def run_model_training(dataset_file, models_folder, plots_folder):
    _, X_train, y_train, X_val, y_val =  get_dataset(dataset_file)

    input_dimensions = X_train[0].shape
    model = create_model(input_dimensions)

    training_history, model = training(model, X_train, y_train, X_val, y_val)

    model_name = f"model_{datetime.now():%Y%m%d_%H%M}"
    model_path = os.path.join(models_folder, model_name + ".h5")
    model.save(model_path, save_format="h5")

    plot_training_history(training_history, model_name, plots_folder)

    return model

def plot_training_history(history, model_name, plots_folder):
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

    plot_path = os.path.join(plots_folder, model_name + "_training.png")
    fig.savefig(plot_path)
    fig.show()

def generate_sequence(model, sequence):
    dim=sequence.shape
    generated_sequence_size = dim[0]
    generated_sequence = np.zeros(dim)

    # initial condition
    generated_sequence[0:WINDOW-1] = sequence[0:WINDOW-1]

    for i in range(generated_sequence_size-WINDOW+1):
        # input = generated_sequence[i:i+WINDOW-1]
        input = sequence[i:i+WINDOW-1]
        predicted_frame = model.predict(np.expand_dims(input, axis=0), verbose=0)[1][0]
        generated_sequence[i+WINDOW-1] = predicted_frame
    
    return generated_sequence

def generate_sequences(model, DATASET_FILE, generated_data_file):
    dataset, _, _, _, _ = get_dataset(DATASET_FILE, split=False)
    
    generated_dataset = np.zeros(dataset.shape)

    for i, sequence in enumerate(dataset):
        generated_sequence = generate_sequence(model, sequence)
        generated_dataset[i,:] = generated_sequence

    np.save(generated_data_file, generated_dataset)

    return generated_dataset

def load_generated_sequences(generated_data_file):
    dataset = np.load(generated_data_file)
    return dataset

    
    with open(model_stats_file, 'w') as f:
        f.write(f"sequence_id, model_duration, sequence_error\n")

        for sequence_id, sequence in enumerate(dataset):

            start_time = datetime.now()
            generated_sequence = generate_sequence(model, sequence, window)
            end_time = datetime.now()
            model_duration = (end_time - start_time).total_seconds()
            
            sequence_errors = []
            for frame, generated_frame in zip(sequence, generated_sequence):
                frame_error = np.square(frame - generated_frame)
                sequence_errors.append(frame_error)
            sequence_error = np.average(sequence_errors)

            f.write(f"{sequence_id}, {model_duration}, {sequence_error}\n")
