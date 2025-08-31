import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, Dropout, LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential

from config import PredictorConfig, Threshold


def create_loss_function(w_normal, w_hypo, w_hyper):
    def custom_loss(y_true, y_pred):
        mae = K.abs(y_pred - y_true)

        normal_abs_error = mae * tf.cast(tf.logical_and(y_true >= Threshold.HYPOGLYCEMIA, y_true <= Threshold.HYPERGLYCEMIA), K.floatx()) * w_normal
        penalty_lower = K.cast(y_true < Threshold.HYPOGLYCEMIA, K.floatx()) * mae * w_hypo
        penalty_upper = K.cast(y_true > Threshold.HYPERGLYCEMIA, K.floatx()) * mae * w_hyper

        return K.mean(normal_abs_error + penalty_lower + penalty_upper)

    return custom_loss

class Predictor:
    def __init__(self):
        self.model = None
        self.history = None

        self.model = Sequential([Conv1D(filters=32, kernel_size=4, input_shape=(PredictorConfig.TRAIN_WINDOW_SIZE, 6)),
                                 Dropout(0.1),
                                 Conv1D(filters=16, kernel_size=4),
                                 Dropout(0.1),
                                 Conv1D(filters=8, kernel_size=4),
                                 Dropout(0.4),
                                 LSTM(units=8, return_sequences=True),
                                 Dropout(0.1),
                                 Flatten(),
                                 Dense(PredictorConfig.N_PREDICTION, activation=PredictorConfig.ACTIVATION),
                                 Dense(units=PredictorConfig.N_PREDICTION)])

        weights = PredictorConfig.WEIGHTS
        loss_func = create_loss_function(*weights)
        self.model.compile(optimizer=PredictorConfig.OPTIMIZER, loss=loss_func)

    def train(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs, batch_size):
        print("\nTraining CGM predictor...\n")
        self.history = self.model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, validation_data=(X_val_seq, y_val_seq))
        print("\nCGM predictor is trained successfully!")

    def predict(self, X_test_seq):
        print("\nPredicting next CGM sequence...")
        y_pred = self.model.predict(X_test_seq)
        return y_pred
