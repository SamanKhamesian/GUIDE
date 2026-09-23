import random

import numpy as np
import pandas as pd
import tensorflow as tf

from config import DataConfig, PredictorConfig
from predictor import Predictor
from preprocess import DataController


PATIENT_IDS = DataConfig.PATIENTS
SEED = 42

ROLLOUT_HOURS = 24
MAX_ROLLOUTS_PER_PATIENT = 10

STEPS_PER_HOUR = PredictorConfig.N_PREDICTION
CONTEXT_STEPS = PredictorConfig.TRAIN_WINDOW_SIZE
ROLLOUT_STEPS = ROLLOUT_HOURS * STEPS_PER_HOUR
TOTAL_SEQUENCE_STEPS = CONTEXT_STEPS + ROLLOUT_STEPS

HOUR_INDEX = 0
SLEEP_INDEX = 1
TIME_SINCE_MEAL_INDEX = 2
TIME_SINCE_INSULIN_INDEX = 3
CARB_INDEX = 4
BOLUS_INDEX = 5
BASAL_INDEX = 6
CGM_INDEX = 7
PREDICTOR_FEATURE_INDEX = 4

PREDICTIONS_FILE = 'glimmer_24h_rollout_predictions.csv'
ROLLOUT_METRICS_FILE = 'glimmer_24h_rollout_metrics.csv'
PATIENT_METRICS_FILE = 'glimmer_24h_patient_metrics.csv'
COHORT_SUMMARY_FILE = 'glimmer_24h_cohort_summary.csv'


def set_seed(seed):
    """
    Set the single reproducibility seed used for model training.
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)


class RecursiveRolloutExperiment:
    def __init__(self, dataset_name, patient_id, seed, max_rollouts):
        self.dataset_name = dataset_name
        self.patient_id = patient_id
        self.seed = seed
        self.max_rollouts = max_rollouts

        tf.keras.backend.clear_session()
        set_seed(seed)

        self.data = DataController(dataset_name, patient_id)
        self.predictor = Predictor()

        self.reserved_timestamps = pd.to_datetime(self.data.X_reserved[:, 0])
        self.reserved_real = np.asarray(self.data.X_reserved[:, 1:], dtype=float)
        self.reserved_scaled = self.data.get_transform(self.reserved_real)

        self._validate_setup()

    def _validate_setup(self):
        """
        Validate feature dimensions and rollout settings.
        """

        if ROLLOUT_HOURS <= 0:
            raise ValueError('ROLLOUT_HOURS must be positive.')

        if STEPS_PER_HOUR != 12:
            raise ValueError('This experiment expects 12 five-minute predictions per hour.')

        if CONTEXT_STEPS != 72:
            raise ValueError('This experiment expects a six-hour, 72-step GLIMMER input window.')

        if self.reserved_real.ndim != 2 or self.reserved_real.shape[1] != 10:
            raise ValueError(f'Expected held-out data with shape (n, 10), received {self.reserved_real.shape}.')

        if len(self.reserved_real) < TOTAL_SEQUENCE_STEPS:
            raise ValueError(f'Patient {self.patient_id} does not have enough held-out data for one 24-hour rollout.')

    def train(self):
        """
        Train one patient-specific GLIMMER model with the single fixed seed.
        """

        self.predictor.train(self.data.X_predictor_train, self.data.y_predictor_train, self.data.X_predictor_val, self.data.y_predictor_val, epochs=PredictorConfig.EPOCHS, batch_size=PredictorConfig.BATCH_SIZE)

    def get_rollout_starts(self):
        """
        Select deterministic rollouts with non-overlapping 24-hour targets.
        """

        max_start = len(self.reserved_real) - TOTAL_SEQUENCE_STEPS
        candidate_starts = list(range(0, max_start + 1, STEPS_PER_HOUR))

        selected_starts = []
        previous_target_end = -1

        for start_index in candidate_starts:
            target_start = start_index + CONTEXT_STEPS

            if target_start < previous_target_end:
                continue

            selected_starts.append(start_index)
            previous_target_end = target_start + ROLLOUT_STEPS

            if self.max_rollouts is not None and len(selected_starts) >= self.max_rollouts:
                break

        if not selected_starts:
            raise ValueError(f'Patient {self.patient_id} has insufficient held-out rows for rollout evaluation.')

        return selected_starts

    def _validate_committed_inputs(self, actual_block):
        """
        Confirm that all recorded non-CGM inputs were replayed correctly.
        """

        committed_real = self.data.get_inverse_transform(self.data.X[0])[-STEPS_PER_HOUR:]

        if not np.allclose(committed_real[:, TIME_SINCE_MEAL_INDEX:CGM_INDEX], actual_block[:, TIME_SINCE_MEAL_INDEX:CGM_INDEX], rtol=0.0, atol=1e-5):
            raise ValueError(f'Replayed inputs do not match the held-out sequence for patient {self.patient_id}.')

    def _commit_prediction(self, predicted_cgm, actual_block):
        """
        Feed predicted CGM back while replaying the recorded future inputs.
        """

        self.data.commit_shift(predicted_cgm, actual_block[:, BASAL_INDEX], actual_block[:, BOLUS_INDEX], actual_block[:, TIME_SINCE_INSULIN_INDEX], actual_block[:, CARB_INDEX], actual_block[:, TIME_SINCE_MEAL_INDEX])
        self._validate_committed_inputs(actual_block)

    def _create_records(self, rollout_id, start_index, rollout_hour, actual_block, predicted_cgm):
        """
        Create point-level records for one predicted hour.
        """

        records = []
        context_start_time = self.reserved_timestamps[start_index]
        target_start_index = start_index + CONTEXT_STEPS
        target_start_time = self.reserved_timestamps[target_start_index]
        block_start_index = target_start_index + (rollout_hour - 1) * STEPS_PER_HOUR

        for step_in_hour, (actual_cgm, predicted_value) in enumerate(zip(actual_block[:, CGM_INDEX], predicted_cgm), start=1):
            data_index = block_start_index + step_in_hour - 1
            horizon_step = (rollout_hour - 1) * STEPS_PER_HOUR + step_in_hour
            error = predicted_value - actual_cgm

            records.append({'seed': self.seed,
                            'patient_id': self.patient_id,
                            'rollout_id': rollout_id,
                            'start_index': start_index,
                            'context_start_time': context_start_time,
                            'target_start_time': target_start_time,
                            'timestamp': self.reserved_timestamps[data_index],
                            'rollout_hour': rollout_hour,
                            'step_in_hour': step_in_hour,
                            'horizon_step': horizon_step,
                            'horizon_minutes': horizon_step * 5,
                            'hour': actual_block[step_in_hour - 1, HOUR_INDEX],
                            'sleep': actual_block[step_in_hour - 1, SLEEP_INDEX],
                            'time_since_last_meal': actual_block[step_in_hour - 1, TIME_SINCE_MEAL_INDEX],
                            'time_since_last_insulin': actual_block[step_in_hour - 1, TIME_SINCE_INSULIN_INDEX],
                            'carb': actual_block[step_in_hour - 1, CARB_INDEX],
                            'bolus': actual_block[step_in_hour - 1, BOLUS_INDEX],
                            'basal': actual_block[step_in_hour - 1, BASAL_INDEX],
                            'actual_cgm': actual_cgm,
                            'predicted_cgm': predicted_value,
                            'error': error,
                            'absolute_error': abs(error),
                            'squared_error': error ** 2})

        return records

    def collect_predictions(self):
        """
        Generate recursive predictions for all selected held-out rollouts.
        """

        records = []
        rollout_starts = self.get_rollout_starts()

        print(f'\nRunning {len(rollout_starts)} recursive rollouts for patient {self.patient_id}...')

        for rollout_id, start_index in enumerate(rollout_starts):
            initial_window = self.reserved_scaled[start_index:start_index + CONTEXT_STEPS]
            self.data.X = initial_window[None, :, :].copy()

            for rollout_hour in range(1, ROLLOUT_HOURS + 1):
                actual_start = start_index + CONTEXT_STEPS + (rollout_hour - 1) * STEPS_PER_HOUR
                actual_block = self.reserved_real[actual_start:actual_start + STEPS_PER_HOUR]
                predicted_cgm = self.predictor.predict(self.data.X[:, :, PREDICTOR_FEATURE_INDEX:])[0]

                records.extend(self._create_records(rollout_id, start_index, rollout_hour, actual_block, predicted_cgm))
                self._commit_prediction(predicted_cgm, actual_block)

            print(f'Completed rollout {rollout_id + 1}/{len(rollout_starts)}')

        return pd.DataFrame(records)


def calculate_group_metrics(df, group_columns):
    """
    Calculate hourly and cumulative errors for the requested grouping.
    """

    df_metrics = df.groupby(group_columns + ['rollout_hour'], as_index=False).agg(num_points=('error', 'size'), sum_absolute_error=('absolute_error', 'sum'), sum_squared_error=('squared_error', 'sum'), sum_error=('error', 'sum'))
    df_metrics = df_metrics.sort_values(group_columns + ['rollout_hour']).reset_index(drop=True)

    df_metrics['mae'] = df_metrics['sum_absolute_error'] / df_metrics['num_points']
    df_metrics['rmse'] = np.sqrt(df_metrics['sum_squared_error'] / df_metrics['num_points'])
    df_metrics['bias'] = df_metrics['sum_error'] / df_metrics['num_points']

    grouped = df_metrics.groupby(group_columns, sort=False)
    df_metrics['cumulative_points'] = grouped['num_points'].cumsum()
    df_metrics['cumulative_absolute_error'] = grouped['sum_absolute_error'].cumsum()
    df_metrics['cumulative_squared_error'] = grouped['sum_squared_error'].cumsum()
    df_metrics['cumulative_error'] = grouped['sum_error'].cumsum()

    df_metrics['cumulative_mae'] = df_metrics['cumulative_absolute_error'] / df_metrics['cumulative_points']
    df_metrics['cumulative_rmse'] = np.sqrt(df_metrics['cumulative_squared_error'] / df_metrics['cumulative_points'])
    df_metrics['cumulative_bias'] = df_metrics['cumulative_error'] / df_metrics['cumulative_points']

    output_columns = group_columns + ['rollout_hour', 'num_points', 'mae', 'rmse', 'bias', 'cumulative_points', 'cumulative_mae', 'cumulative_rmse', 'cumulative_bias']
    return df_metrics[output_columns]


def calculate_cohort_summary(df_patient):
    """
    Summarize hourly error across patient-level results.
    """

    metric_columns = ['mae', 'rmse', 'bias', 'cumulative_mae', 'cumulative_rmse', 'cumulative_bias']
    summary_frames = []

    for metric in metric_columns:
        df_metric = df_patient.groupby('rollout_hour')[metric].agg(['mean', 'std']).reset_index()
        df_metric = df_metric.rename(columns={'mean': f'{metric}_mean', 'std': f'{metric}_sd'})
        summary_frames.append(df_metric)

    df_summary = summary_frames[0]

    for df_metric in summary_frames[1:]:
        df_summary = df_summary.merge(df_metric, on='rollout_hour', how='inner')

    num_patients = df_patient.groupby('rollout_hour')['patient_id'].nunique().rename('num_patients').reset_index()
    df_summary = df_summary.merge(num_patients, on='rollout_hour', how='left')
    return df_summary


def validate_predictions(df):
    """
    Validate the saved point-level rollout predictions.
    """

    required_columns = ['seed', 'patient_id', 'rollout_id', 'start_index', 'timestamp', 'rollout_hour', 'step_in_hour', 'horizon_step', 'horizon_minutes', 'actual_cgm', 'predicted_cgm', 'error', 'absolute_error', 'squared_error']
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f'Missing rollout columns: {missing_columns}')

    duplicated = df.duplicated(['patient_id', 'rollout_id', 'horizon_step']).sum()

    if duplicated > 0:
        raise ValueError(f'Found {duplicated} duplicated rollout prediction records.')

    points_per_hour = df.groupby(['patient_id', 'rollout_id', 'rollout_hour']).size()

    if not (points_per_hour == STEPS_PER_HOUR).all():
        raise ValueError('Every rollout hour must contain exactly 12 predictions.')

    hours_per_rollout = df.groupby(['patient_id', 'rollout_id'])['rollout_hour'].nunique()

    if not (hours_per_rollout == ROLLOUT_HOURS).all():
        raise ValueError('Every rollout must contain exactly 24 predicted hours.')


def save_error_outputs(df):
    """
    Validate and save all error summaries needed for later reporting.
    """

    validate_predictions(df)
    df.to_csv(PREDICTIONS_FILE, index=False)

    df_rollout = calculate_group_metrics(df, ['patient_id', 'rollout_id'])
    df_rollout.to_csv(ROLLOUT_METRICS_FILE, index=False)

    df_patient = calculate_group_metrics(df, ['patient_id'])
    df_patient.to_csv(PATIENT_METRICS_FILE, index=False)

    df_summary = calculate_cohort_summary(df_patient)
    df_summary.to_csv(COHORT_SUMMARY_FILE, index=False)

    print(f'\nSaved {len(df)} point-level predictions to {PREDICTIONS_FILE}')
    print(f'Saved per-rollout hourly errors to {ROLLOUT_METRICS_FILE}')
    print(f'Saved patient-level hourly errors to {PATIENT_METRICS_FILE}')
    print(f'Saved cohort hourly error summary to {COHORT_SUMMARY_FILE}')
    return df_rollout, df_patient, df_summary


def run_experiment():
    """
    Train one GLIMMER model per patient and run the 24-hour evaluation.
    """

    all_results = []

    for patient_id in PATIENT_IDS:
        experiment = RecursiveRolloutExperiment(dataset_name=DataConfig.DATASET, patient_id=patient_id, seed=SEED, max_rollouts=MAX_ROLLOUTS_PER_PATIENT)
        experiment.train()

        df_patient = experiment.collect_predictions()
        all_results.append(df_patient)

    df = pd.concat(all_results, ignore_index=True)
    save_error_outputs(df)
    return df


def read_predictions():
    """
    Read and validate saved point-level rollout predictions.
    """

    df = pd.read_csv(PREDICTIONS_FILE)
    validate_predictions(df)

    num_patients = df['patient_id'].nunique()
    num_rollouts = df[['patient_id', 'rollout_id']].drop_duplicates().shape[0]

    print(f'\nLoaded {len(df)} point-level predictions from {PREDICTIONS_FILE}')
    print(f'Patients: {num_patients}')
    print(f'Recursive 24-hour rollouts: {num_rollouts}')
    return df


def print_key_horizons(df_summary):
    """
    Print compact error results at selected rollout horizons.
    """

    selected_hours = [1, 6, 12, 18, 24]
    df_table = df_summary[df_summary['rollout_hour'].isin(selected_hours)].copy()

    df_table['Hourly MAE'] = df_table.apply(lambda row: f'{row["mae_mean"]:.2f} +/- {row["mae_sd"]:.2f}', axis=1)
    df_table['Hourly RMSE'] = df_table.apply(lambda row: f'{row["rmse_mean"]:.2f} +/- {row["rmse_sd"]:.2f}', axis=1)
    df_table['Cumulative MAE'] = df_table.apply(lambda row: f'{row["cumulative_mae_mean"]:.2f} +/- {row["cumulative_mae_sd"]:.2f}', axis=1)
    df_table['Cumulative RMSE'] = df_table.apply(lambda row: f'{row["cumulative_rmse_mean"]:.2f} +/- {row["cumulative_rmse_sd"]:.2f}', axis=1)
    df_table = df_table[['rollout_hour', 'Hourly MAE', 'Hourly RMSE', 'Cumulative MAE', 'Cumulative RMSE', 'num_patients']]

    print('\nRecursive rollout error at selected horizons')
    print(df_table.to_string(index=False))
    print('\nValues are cohort mean +/- SD across patient-level errors.')
    return df_table


def main():
    """
    Rebuild saved error summaries without retraining GLIMMER.
    """

    df = read_predictions()
    _, _, df_summary = save_error_outputs(df)
    print_key_horizons(df_summary)


if __name__ == '__main__':
    # Run this once to train one patient-specific model and generate 24-hour rollouts:
    # run_experiment()

    # After glimmer_24h_rollout_predictions.csv has been generated, rebuild the summaries:
    main()