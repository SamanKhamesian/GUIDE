import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import PredictorConfig, Threshold, SleepTime, TD3Config


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    @staticmethod
    def __calculate_moving_average(data, window_size):
        window_size = int(window_size)
        moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        return np.concatenate((np.zeros(window_size - 1), moving_avg), axis=0)


    @staticmethod
    def __compute_time_since_last_event(event_series, init_value=288):
        time_since = []
        last_event_idx = -init_value

        for i, value in enumerate(event_series):
            if value > 0:
                last_event_idx = i
            time_since.append(i - last_event_idx)

        return np.array(time_since, dtype=int)

    def create_input_features(self, dataset_name, patient_id):
        if dataset_name == "ohio":
            df_train = pd.read_csv(f'./dataset/OhioT1DM/{patient_id}_train.csv')
            df_test = pd.read_csv(f'./dataset/OhioT1DM/{patient_id}_test.csv')

            glucose_col = "glucose"
            carb_col = "carbs"
            bolus_col = "bolus"
            basal_col = "basal"
            timestamp_col = "index"

        elif dataset_name == "azt1d":
            dataframe = pd.read_csv(f'./dataset/AZT1D/Subject {patient_id}.csv')
            split_index = int((1 - PredictorConfig.SPLIT_RATIO) * len(dataframe))
            df_train = dataframe.iloc[:split_index].copy()
            df_test = dataframe.iloc[split_index:].copy()

            glucose_col = "CGM"
            carb_col = "CarbSize"
            bolus_col = "TotalBolusInsulinDelivered"
            basal_col = "Basal"
            timestamp_col = "EventDateTime"

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Clean and reset
        df_train.replace(-1, np.nan, inplace=True)
        df_test.replace(-1, np.nan, inplace=True)

        df_train = df_train.dropna(subset=[glucose_col])
        df_test = df_test.dropna(subset=[glucose_col])

        df_train.fillna(0, inplace=True)
        df_test.fillna(0, inplace=True)

        df_train = df_train.reset_index()
        df_test = df_test.reset_index()

        # Hour
        df_train["timestamp"] = pd.to_datetime(df_train[timestamp_col])
        df_train["hour"] = df_train["timestamp"].dt.hour

        df_test["timestamp"] = pd.to_datetime(df_test[timestamp_col])
        df_test["hour"] = df_test["timestamp"].dt.hour

        # Sleep
        df_train["sleep"] = np.where((df_train["hour"] >= SleepTime.BED_TIME) | (df_train["hour"] < SleepTime.WAKE_UP), 1.0, 0.0)
        df_test["sleep"] = np.where((df_test["hour"] >= SleepTime.BED_TIME) | (df_test["hour"] < SleepTime.WAKE_UP), 1.0, 0.0)

        # Time since the last meal
        df_train["time_since_last_meal"] = self.__compute_time_since_last_event(df_train[carb_col].values)
        df_test["time_since_last_meal"] = self.__compute_time_since_last_event(df_test[carb_col].values)

        # Time since the last insulin injection
        df_train["time_since_last_insulin"] = self.__compute_time_since_last_event(df_train[bolus_col].values)
        df_test["time_since_last_insulin"] = self.__compute_time_since_last_event(df_test[bolus_col].values)

        relevant_features = ['hour', 'sleep', 'time_since_last_meal', 'time_since_last_insulin', carb_col, bolus_col, basal_col, glucose_col]

        # moving average
        moving_avg_200 = self.__calculate_moving_average(df_train[glucose_col].values, PredictorConfig.MA_WINDOW_SIZE)
        df_train['glucose_MA_200'] = moving_avg_200
        df_train = df_train[df_train['glucose_MA_200'] != 0.0]
        df_train = df_train.reset_index(drop=True)

        moving_avg_200 = self.__calculate_moving_average(df_test[glucose_col].values, PredictorConfig.MA_WINDOW_SIZE)
        df_test['glucose_MA_200'] = moving_avg_200
        df_test = df_test[df_test['glucose_MA_200'] != 0.0]
        df_test = df_test.reset_index(drop=True)

        relevant_features = relevant_features + ['glucose_MA_200']

        # glucose class
        choices = [0, 1, 2]

        conditions = [df_train[glucose_col] < Threshold.HYPOGLYCEMIA,
                      (df_train[glucose_col] >= Threshold.HYPOGLYCEMIA) & (df_train[glucose_col] <= Threshold.HYPERGLYCEMIA),
                      df_train[glucose_col] > Threshold.HYPERGLYCEMIA]
        df_train['glucose_class'] = np.select(conditions, choices)

        conditions = [df_test[glucose_col] < Threshold.HYPOGLYCEMIA,
                      (df_test[glucose_col] >= Threshold.HYPOGLYCEMIA) & (df_test[glucose_col] <= Threshold.HYPERGLYCEMIA),
                      df_test[glucose_col] > Threshold.HYPERGLYCEMIA]
        df_test['glucose_class'] = np.select(conditions, choices)

        relevant_features = relevant_features + ['glucose_class']

        return (df_train[relevant_features].values,
                df_train[glucose_col].values,
                df_test[relevant_features].values,
                df_test[glucose_col].values)


    @staticmethod
    def __create_x_y_sequences(X, y, time_steps, prediction_horizon, shift=1):
        X_seq, y_seq = [], []
        for i in range(0, len(X) - time_steps - prediction_horizon + 1, shift):
            X_seq.append(X[i:i + time_steps])
            if prediction_horizon > 0:
                y_temp = y[i + time_steps:i + time_steps + prediction_horizon]
                y_seq.append(y_temp)
        return np.array(X_seq), np.array(y_seq)


    def create_train_val_data(self, _X_train_val_, _y_train_val_):
        _X_train_, _X_val_, _y_train_, _y_val_ = train_test_split(_X_train_val_, _y_train_val_, test_size=PredictorConfig.SPLIT_RATIO, shuffle=False)

        _X_train_ = self.scaler.fit_transform(_X_train_)[:, 4:]
        _X_val_ = self.scaler.transform(_X_val_)[:, 4:]

        _X_train_seq_, _y_train_seq_ = self.__create_x_y_sequences(_X_train_, _y_train_, PredictorConfig.TRAIN_WINDOW_SIZE, PredictorConfig.N_PREDICTION)
        _X_val_seq_, _y_val_seq_ = self.__create_x_y_sequences(_X_val_, _y_val_, PredictorConfig.TRAIN_WINDOW_SIZE, PredictorConfig.N_PREDICTION)

        return _X_train_seq_, _y_train_seq_, _X_val_seq_, _y_val_seq_


    def create_rl_train_test_data(self, X, y):
        x = self.scaler.transform(X)

        _X_train_, _X_test_, _y_train_, _y_test_ = train_test_split(x, y, test_size=0.2, shuffle=False)
        _X_train_seq_, _ = self.__create_x_y_sequences(X=_X_train_, y=_y_train_, time_steps=6 * 12, prediction_horizon=0, shift=1)
        _X_test_seq_, _y_test_seq_ = self.__create_x_y_sequences(X=_X_test_, y=_y_test_, time_steps=6 * 12, prediction_horizon=6 * 12 * 4, shift=12)

        return _X_train_seq_, _X_test_seq_, _y_train_, _y_test_seq_


class DataController:
    def __init__(self, dataset_name, patient_id):
        self.dataset_name = dataset_name
        self.patient_id = patient_id

        self.__preprocessor = Preprocessor()

        X_train, y_train, X_test, y_test = self.__preprocessor.create_input_features(dataset_name=dataset_name, patient_id=patient_id)
        self.X_predictor_train, self.y_predictor_train, self.X_predictor_val, self.y_predictor_val = self.__preprocessor.create_train_val_data(X_train, y_train)

        self.X_rl_train, self.X_rl_test, self._y_rl_train_, self.y_rl_test = self.__preprocessor.create_rl_train_test_data(X_test, y_test)

        self.X = self.X_rl_train[0][None, :, :]

    def get_inverse_transform(self, data):
        return self.__preprocessor.scaler.inverse_transform(data)

    def get_transform(self, data):
        return self.__preprocessor.scaler.transform(data)

    @staticmethod
    def __next_moving_average(current_MA_200, new_data):
        alpha = 2 / (PredictorConfig.MA_WINDOW_SIZE + 1)
        EMA_old = current_MA_200
        EMA_new = (new_data - EMA_old) * alpha + EMA_old
        return EMA_new

    def commit_shift(self, predicted_cgm, bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array):
        # Step 1: Get the current unscaled window
        original_real = self.get_inverse_transform(self.X[0])  # shape: (72, 10)

        # Step 2: Shift → remove first 12 rows
        original_real = original_real[PredictorConfig.N_PREDICTION:]  # shape: (60, 10)

        future_12_source = original_real[-12:]

        # Step 3: Prepare 12 new rows (from predicted CGM + inputs)
        hour_array = np.round(future_12_source[-12:, 0]).astype(int)
        basal_array = future_12_source[-12:, 6].copy()

        # Count how many times the last hour appears
        last_hour = hour_array[-1]
        count = np.sum(hour_array == last_hour)
        remaining = 12 - count

        # Generate next 12-hour values
        next_12_hours = [last_hour] * remaining + [(last_hour + 1) % 24] * (12 - remaining)
        next_12_sleep_flags = [1.0 if (SleepTime.BED_TIME <= h <= SleepTime.WAKE_UP) else 0.0 for h in next_12_hours]

        # MA calculation
        new_MA_input = []
        current_MA_input = future_12_source[-1, 8]  # last known MA
        for cgm_data in predicted_cgm:
            new_MA = self.__next_moving_average(current_MA_input, cgm_data)
            new_MA_input.append(new_MA)
            current_MA_input = new_MA

        # Glucose class
        glucose_class = [0 if val < Threshold.HYPOGLYCEMIA else 1 if val <= Threshold.HYPERGLYCEMIA else 2 for val in predicted_cgm]

        # Step 4: Build new 12-row block
        new_block = np.column_stack([next_12_hours,
                                     next_12_sleep_flags,
                                     time_since_last_meal_array,
                                     time_since_last_injection_array,
                                     carb_array,
                                     bolus_array,
                                     basal_array,
                                     predicted_cgm,
                                     new_MA_input,
                                     glucose_class])  # shape: (12, 10)

        # Step 5: Concatenate shifted + new
        final_real = np.vstack([original_real, new_block])  # shape: (72, 10)

        # Step 6: Scale and update
        final_scaled = self.get_transform(final_real)
        self.X[0] = final_scaled

