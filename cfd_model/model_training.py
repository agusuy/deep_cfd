from datetime import datetime
from tensorflow.distribute import MirroredStrategy
from keras.models import Model
from keras.layers import Input, Conv2D, ConvLSTM2D, MaxPool3D, UpSampling3D, LeakyReLU
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.optimizers import Adam


def create_model(input_size):
    GPUS = ["GPU:0","GPU:1"]
    # GPUS = ["GPU:0"]
    strategy = MirroredStrategy(GPUS)

    with strategy.scope():
        input_dimensions = input_size

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

    model.summary()
    return model



def training(model, X_train, y_train, X_val, y_val):
    batch_size = 16

    # epochs = 4
    epochs = 400

    history = model.fit(
        X_train, [X_train, y_train],
        # validation_data=(X_val, [X_val, y_val]),
        batch_size=batch_size, epochs=epochs,
        verbose=1)

    model_name = f"./models/_model_{datetime.now():%Y%m%d_%H%M%S}"
    model.save(model_name + ".h5", save_format="h5")

    return history, model
