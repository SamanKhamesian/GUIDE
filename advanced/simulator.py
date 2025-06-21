from config import Action, TargetModel, PredictorConfig
from predictor import Predictor
from preprocess import DataController
import numpy as np

class Simulator:
    def __init__(self, patient_id):
        self.data = DataController(patient_id, TargetModel.GLUCOSE)
        self.predictor = Predictor()

    def train(self):
        self.predictor.train(self.data.X_train_seq_,
                             self.data.y_train_seq_,
                             self.data.X_val_seq_,
                             self.data.y_val_seq_,
                             epochs=PredictorConfig.EPOCHS,
                             batch_size=PredictorConfig.BATCH_SIZE)

    def predict_next_cgm(self):
        return self.predictor.predict(self.data.X_test[:, :, 4:])[0]

    def get_full_current_window(self):
        return self.data.X_test[0].copy()

    def get_time_window(self):
        full_real = self.data.get_inverse_transform(self.data.X_test[0])
        return full_real[-12:, 0]

    def get_sleep_mode(self):
        full_real = self.data.get_inverse_transform(self.data.X_test[0])
        return bool(full_real[-1, 1])

    def apply_action_to_inputs(self, full_current_window, action):
        action_type, value, time_index = action
        time_index = int(time_index)

        # Step 1: Inverse transform
        full_real = self.data.get_inverse_transform(full_current_window)

        # Step 2: Get previous time-since values (from last row)
        time_since_last_meal = full_real[-1, 2]
        time_since_last_injection = full_real[-1, 3]

        # Step 3: Initialize zero lists
        carb_array = [0.0] * 12
        bolus_array = [0.0] * 12

        # Step 4: Initialize time-since arrays
        time_since_last_meal_array = [time_since_last_meal + i + 1 for i in range(12)]
        time_since_last_injection_array = [time_since_last_injection + i + 1 for i in range(12)]

        # Step 5: Apply action
        if action_type == Action.EAT:
            carb_array[time_index] = value
            for i in range(time_index, 12):
                time_since_last_meal_array[i] = i - time_index

        elif action_type == Action.INJECT:
            bolus_array[time_index] = value
            for i in range(time_index, 12):
                time_since_last_injection_array[i] = i - time_index

        return (bolus_array, [time_since_last_injection] + time_since_last_injection_array,
                carb_array, [time_since_last_meal] + time_since_last_meal_array)

    def commit_next_input(self, predicted_cgm, bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array):
        self.data.commit_shift(predicted_cgm, bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array)

    def reset(self):
        print("\nResetting simulator...")
        self.data = DataController(self.data.patient_id, TargetModel.GLUCOSE)
