import numpy as np

from config import Action, PredictorConfig
from predictor import Predictor
from preprocess import DataController


class Simulator:
    def __init__(self, dataset_name, patient_id):
        self.data = DataController(dataset_name, patient_id)
        self.predictor = Predictor()

    def train(self):
        self.predictor.train(self.data.X_predictor_train,
                             self.data.y_predictor_train,
                             self.data.X_predictor_val,
                             self.data.y_predictor_val,
                             epochs=PredictorConfig.EPOCHS,
                             batch_size=PredictorConfig.BATCH_SIZE)

    def predict_next_cgm(self):
        return self.predictor.predict(self.data.X[:, :, 4:])[0]

    def get_full_current_window(self):
        return self.data.X[0].copy()

    def get_time_window(self):
        full_real = self.data.get_inverse_transform(self.data.X[0])
        return np.round(full_real[-12:, 0]).astype(int)

    def get_sleep_mode(self):
        full_real = self.data.get_inverse_transform(self.data.X[0])
        return bool(round(full_real[-1, 1]))

    @staticmethod
    def select_main_meal_hours():
        breakfast_hour = np.random.choice([7, 8, 9])
        lunch_hour = np.random.choice([12, 13, 14])
        dinner_hour = np.random.choice([19, 20, 21, 22])
        return [breakfast_hour, lunch_hour, dinner_hour]

    @staticmethod
    def select_main_meal_portion(low, high, mean, sd):
        while True:
            x = np.random.normal(mean, sd)
            if low <= x <= high:
                return x

    @staticmethod
    def heuristic_basal_controller(predicted_cgm):
        """
        Heuristic controller for basal insulin using piecewise linear logic:
        - 0.0 U/hr if predicted CGM < 70
        - Linear increase from 0.0 to 1.0 between CGM 70–100
        - Flat 1.0 U/hr between CGM 100–180
        - Linear increase from 1.0 to 2.0 between CGM 180–250
        - Clipped to 2.0 U/hr for CGM > 250
        """

        avg_cgm = np.mean(predicted_cgm)

        if avg_cgm <= 70:
            return 0.0
        elif avg_cgm <= 100:
            return np.interp(avg_cgm, [70, 100], [0.0, 1.0])
        elif avg_cgm <= 180:
            return 1.0
        elif avg_cgm <= 250:
            return np.interp(avg_cgm, [180, 250], [1.0, 2.0])
        else:
            return 2.0

    def apply_action_to_inputs(self, full_current_window, action, main_meal_action=None):
        action_type, value, time_index = action
        time_index = int(time_index)

        # Step 1: Inverse transform
        full_real = self.data.get_inverse_transform(full_current_window)

        # Step 2: Get previous time-since values (from last row)
        time_since_last_meal = int(round(full_real[-1, 2]))
        time_since_last_injection = int(round(full_real[-1, 3]))

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

        # Step 6: Apply the main meal action
        if main_meal_action is not None:
            _, m_value, m_time_index = main_meal_action
            m_time_index = int(m_time_index)
            carb_array[m_time_index] += m_value

            meal_indices = [m_time_index]

            if action_type == Action.EAT:
                meal_indices.append(time_index)

            meal_indices = sorted(set(meal_indices))

            for idx in meal_indices:
                for j in range(idx, 12):
                    time_since_last_meal_array[j] = j - idx

        return (bolus_array, [time_since_last_injection] + time_since_last_injection_array,
                carb_array, [time_since_last_meal] + time_since_last_meal_array)

    def commit_next_input(self, predicted_cgm, basal_array, bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array):
        self.data.commit_shift(predicted_cgm, basal_array, bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array)

    def reset(self, state_index, is_testing=False):
        print("\nResetting simulator...")
        if is_testing:
            self.data.X = self.data.X_rl_test[state_index]
        else:
            self.data.X = self.data.X_rl_train[state_index]

        self.data.X = self.data.X[None, :, :]
