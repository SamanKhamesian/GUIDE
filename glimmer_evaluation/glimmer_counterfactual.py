import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import PchipInterpolator

from config import DataConfig, EnvConfig, PredictorConfig
from predictor import Predictor
from preprocess import DataController

PATIENT_IDS = DataConfig.PATIENTS
SEEDS = [42, 43, 44, 45, 46]
NUM_TEST_STATES = 10
TIME_INDEX = 0

INSULIN_COUNTERFACTUALS = [0, 2, 5, 10]
CARB_COUNTERFACTUALS = [0, 10, 30, 50]

LAST_HOUR_SIZE = 12
CARB_INDEX = 4
BOLUS_INDEX = 5
CGM_INDEX = 7
PREDICTOR_FEATURE_INDEX = 4

OUTPUT_FILE = '../archive/glimmer_cf_seeds.csv'
MEAN_FIGURE_FILE = '../archive/glimmer_cf_cohort_seed_mean.pdf'

PREDICTION_TIMES = np.arange(0, 65, 5)
PREDICTION_COLUMNS = [f'predicted_cgm_{i}' for i in range(12)]


def set_seed(seed):
    """
    Set the random seed before constructing and training each GLIMMER model.
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)

    except AttributeError:
        tf.random.set_seed(seed)


class CounterfactualExperiment:
    def __init__(self, dataset_name, patient_id, seed, num_test_states, time_index):
        self.dataset_name = dataset_name
        self.patient_id = patient_id
        self.seed = seed
        self.num_test_states = num_test_states
        self.time_index = time_index

        tf.keras.backend.clear_session()
        set_seed(seed)

        self.data = DataController(dataset_name, patient_id)
        self.predictor = Predictor()

        self._validate_setup()

    def _validate_setup(self):
        """
        Validate the counterfactual values and the selected time index.
        """

        if not 0 <= self.time_index < LAST_HOUR_SIZE:
            raise ValueError(f'TIME_INDEX must be between 0 and {LAST_HOUR_SIZE - 1}.')

        insulin_values = [value for value in INSULIN_COUNTERFACTUALS if value > 0]
        carb_values = [value for value in CARB_COUNTERFACTUALS if value > 0]

        if not all(EnvConfig.INSULIN_RANGE[0] <= value <= EnvConfig.INSULIN_RANGE[1] for value in insulin_values):
            raise ValueError('Insulin counterfactuals must be within the GUIDE insulin action range.')

        if not all(EnvConfig.CARB_RANGE[0] <= value <= EnvConfig.CARB_RANGE[1] for value in carb_values):
            raise ValueError('Carbohydrate counterfactuals must be within the GUIDE carbohydrate action range.')

    def train(self):
        """
        Train the patient-specific GLIMMER predictor for the selected seed.
        """

        self.predictor.train(X_train_seq=self.data.X_predictor_train,
                             y_train_seq=self.data.y_predictor_train,
                             X_val_seq=self.data.X_predictor_val,
                             y_val_seq=self.data.y_predictor_val,
                             epochs=PredictorConfig.EPOCHS,
                             batch_size=PredictorConfig.BATCH_SIZE)

    def _insert_counterfactual(self, original_real, intervention, value):
        """
        Insert one counterfactual into the most recent hour and scale the window again.
        """

        counterfactual_real = original_real.copy()
        row_index = -LAST_HOUR_SIZE + self.time_index

        if intervention == 'insulin':
            counterfactual_real[row_index, CARB_INDEX] = 0.0
            counterfactual_real[row_index, BOLUS_INDEX] = value

        elif intervention == 'carbohydrate':
            counterfactual_real[row_index, CARB_INDEX] = value
            counterfactual_real[row_index, BOLUS_INDEX] = 0.0
        else:
            raise ValueError(f'Unknown intervention: {intervention}')

        return self.data.get_transform(counterfactual_real)

    def _create_record(self, state_index, intervention, value, original_real, predicted_cgm):
        """
        Create one result record containing the seed, counterfactual, and predicted trajectory.
        """

        row_index = -LAST_HOUR_SIZE + self.time_index
        record = {'seed': self.seed,
                  'patient_id': self.patient_id,
                  'state_index': state_index,
                  'intervention': intervention,
                  'value': value,
                  'time_index': self.time_index,
                  'original_carb': original_real[row_index, CARB_INDEX],
                  'original_bolus': original_real[row_index, BOLUS_INDEX],
                  'current_cgm': original_real[-1, CGM_INDEX]}

        for prediction_index, cgm in enumerate(predicted_cgm):
            record[f'predicted_cgm_{prediction_index}'] = cgm

        return record

    @staticmethod
    def _validate_zero_predictions(metadata, predicted_cgms):
        """
        Confirm that the insulin and carbohydrate zero-action controls are identical.
        """

        insulin_zero_index = metadata.index(('insulin', 0))
        carb_zero_index = metadata.index(('carbohydrate', 0))

        if not np.allclose(predicted_cgms[insulin_zero_index], predicted_cgms[carb_zero_index]):
            raise ValueError('The zero-action counterfactual predictions are not identical.')

    def collect_predictions(self):
        """
        Generate counterfactual predictions for the selected held-out test states.
        """

        records = []
        num_states = min(self.num_test_states, len(self.data.X_rl_test))

        print(f'\nRunning {num_states} test states for patient {self.patient_id}, seed {self.seed}...')

        for state_index in range(num_states):
            original_scaled = self.data.X_rl_test[state_index]
            original_real = self.data.get_inverse_transform(original_scaled)

            counterfactual_windows = []
            metadata = []

            for value in INSULIN_COUNTERFACTUALS:
                counterfactual_scaled = self._insert_counterfactual(original_real, 'insulin', value)
                counterfactual_windows.append(counterfactual_scaled[:, PREDICTOR_FEATURE_INDEX:])
                metadata.append(('insulin', value))

            for value in CARB_COUNTERFACTUALS:
                counterfactual_scaled = self._insert_counterfactual(original_real, 'carbohydrate', value)
                counterfactual_windows.append(counterfactual_scaled[:, PREDICTOR_FEATURE_INDEX:])
                metadata.append(('carbohydrate', value))

            predicted_cgms = self.predictor.predict(np.array(counterfactual_windows))
            self._validate_zero_predictions(metadata, predicted_cgms)

            for (intervention, value), predicted_cgm in zip(metadata, predicted_cgms):
                records.append(self._create_record(state_index,
                                                   intervention,
                                                   value,
                                                   original_real,
                                                   predicted_cgm))

            print(f'Completed test state {state_index + 1}/{num_states}')

        return pd.DataFrame(records)


def run_experiment():
    """
    Train GLIMMER for every patient and seed and save all counterfactual predictions.
    """

    all_results = []

    for seed in SEEDS:
        for patient_id in PATIENT_IDS:
            experiment = CounterfactualExperiment(dataset_name=DataConfig.DATASET,
                                                  patient_id=patient_id,
                                                  seed=seed,
                                                  num_test_states=NUM_TEST_STATES,
                                                  time_index=TIME_INDEX)
            experiment.train()

            df_results = experiment.collect_predictions()
            all_results.append(df_results)

    df_results = pd.concat(all_results, ignore_index=True)
    df_results.to_csv(OUTPUT_FILE, index=False)

    print(f'\nSaved {len(df_results)} counterfactual predictions to {OUTPUT_FILE}')

    return df_results


def read_predictions():
    """
    Read and validate the saved counterfactual predictions.
    """

    df = pd.read_csv(OUTPUT_FILE)

    required_columns = ['seed', 'patient_id', 'state_index', 'intervention', 'value', 'current_cgm'] + PREDICTION_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f'Missing columns in {OUTPUT_FILE}: {missing_columns}')

    duplicated = df.duplicated(['seed', 'patient_id', 'state_index', 'intervention', 'value']).sum()

    if duplicated > 0:
        raise ValueError(f'Found {duplicated} duplicated counterfactual records.')

    num_seeds = df['seed'].nunique()
    num_patients = df['patient_id'].nunique()
    num_states = df[['seed', 'patient_id', 'state_index']].drop_duplicates().shape[0]

    print(f'\nLoaded {len(df)} counterfactual predictions from {OUTPUT_FILE}')
    print(f'Seeds: {num_seeds}')
    print(f'Patients: {num_patients}')
    print(f'Seed-patient-state combinations: {num_states}')

    return df


def get_patient_trajectories(df):
    """
    Average counterfactual trajectories over test states within each patient and seed.
    """

    metadata = df[['seed', 'patient_id', 'state_index', 'intervention', 'value']]
    trajectories = np.column_stack([df['current_cgm'].to_numpy(), df[PREDICTION_COLUMNS].to_numpy()])

    df_long = metadata.loc[metadata.index.repeat(len(PREDICTION_TIMES))].reset_index(drop=True)
    df_long['time'] = np.tile(PREDICTION_TIMES, len(df))
    df_long['cgm'] = trajectories.reshape(-1)

    df_patient = df_long.groupby(['seed', 'patient_id', 'intervention', 'value', 'time'], as_index=False)['cgm'].mean()

    return df_patient


def _style_trajectory_axis(ax):
    """
    Apply the publication style to one trajectory panel.
    """

    ax.set_facecolor('#f7f7f7')
    ax.grid(True, color='#8c8c8c', alpha=0.18, linewidth=0.7)
    ax.tick_params(axis='both', direction='out', width=1.0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.1)


def get_cohort_trajectories(df):
    """
    Calculate one cohort-mean trajectory per training seed.
    """

    df_patient = get_patient_trajectories(df)
    df_seed = df_patient.groupby(['seed', 'intervention', 'value', 'time'], as_index=False)['cgm'].mean()
    return df_seed


def get_seed_summary_trajectories(df):
    """
    Calculate the mean and SD across seed-specific cohort trajectories.
    """

    df_seed = get_cohort_trajectories(df)
    df_summary = df_seed.groupby(['intervention', 'value', 'time'])['cgm'].agg(mean='mean', seed_sd='std').reset_index()
    df_summary['seed_sd'] = df_summary['seed_sd'].fillna(0.0)

    return df_summary


def plot_cohort_trajectories(df, figure_file):
    """
    Plot the mean cohort trajectory with SD across training seeds.
    """

    df_summary = get_seed_summary_trajectories(df)
    time_smooth = np.linspace(PREDICTION_TIMES.min(), PREDICTION_TIMES.max(), 300)

    interventions = ['insulin', 'carbohydrate']
    titles = ['(a) Bolus Insulin Counterfactuals', '(b) Carbohydrate Counterfactuals']

    colors = {'insulin': ['#6b7280', '#90caf9', '#42a5f5', '#1565c0'],
              'carbohydrate': ['#6b7280', '#ffcc80', '#fb8c00', '#d84315']}

    units = {'insulin': 'U', 'carbohydrate': 'g'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=False)

    for ax, intervention, title in zip(axes, interventions, titles):
        df_intervention = df_summary[df_summary['intervention'] == intervention]
        levels = sorted(df_intervention['value'].unique())

        for level, color in zip(levels, colors[intervention]):
            df_level = df_intervention[df_intervention['value'] == level].sort_values('time')
            mean = df_level['mean'].to_numpy()
            seed_sd = df_level['seed_sd'].to_numpy()

            mean_smooth = PchipInterpolator(PREDICTION_TIMES, mean)(time_smooth)
            seed_sd_smooth = PchipInterpolator(PREDICTION_TIMES, seed_sd)(time_smooth)

            ax.fill_between(time_smooth,
                            mean_smooth - seed_sd_smooth,
                            mean_smooth + seed_sd_smooth,
                            color=color,
                            alpha=0.14,
                            linewidth=0)
            ax.plot(time_smooth,
                    mean_smooth,
                    color=color,
                    linewidth=2.2,
                    label=f'{level:g} {units[intervention]}')
            ax.plot(PREDICTION_TIMES, mean, 'o', color=color, markersize=3.8)

        _style_trajectory_axis(ax)

        ax.set_title(title, fontsize=18)
        ax.set_xticks(PREDICTION_TIMES)
        ax.tick_params(axis='both', direction='out', width=1.0, labelsize=16)
        ax.legend(frameon=True, facecolor='white', edgecolor='#bdbdbd', fontsize=14, loc='upper right')

    axes[0].set_ylabel('CGM (mg/dL)', fontsize=18)
    fig.supxlabel('Prediction Horizon (min)', fontsize=18)

    fig.tight_layout()
    fig.savefig(figure_file, dpi=300, bbox_inches='tight')

    print(f'Saved seed-averaged cohort trajectory figure to {figure_file}')
    return fig


def calculate_response_magnitude(df):
    """
    Calculate response magnitudes and variability across patients and seeds.
    """

    patient_seed_summaries = []

    for intervention in ['insulin', 'carbohydrate']:
        df_intervention = df[df['intervention'] == intervention]
        df_reference = df_intervention[df_intervention['value'] == 0].set_index(['seed', 'patient_id', 'state_index'])
        levels = sorted(value for value in df_intervention['value'].unique() if value > 0)

        for level in levels:
            df_level = df_intervention[df_intervention['value'] == level].set_index(['seed', 'patient_id', 'state_index'])
            df_level = df_level.loc[df_reference.index]

            differences = (df_level[PREDICTION_COLUMNS].to_numpy()
                           - df_reference[PREDICTION_COLUMNS].to_numpy())
            mean_differences = differences.mean(axis=1)
            final_differences = differences[:, -1]

            df_state = pd.DataFrame({
                'seed': df_reference.index.get_level_values('seed'),
                'patient_id': df_reference.index.get_level_values('patient_id'),
                'mean_difference': mean_differences,
                'final_difference': final_differences
            })

            df_patient_seed = df_state.groupby(['seed', 'patient_id'], as_index=False)[
                ['mean_difference', 'final_difference']].mean()
            df_patient_seed['intervention'] = intervention
            df_patient_seed['level'] = level
            patient_seed_summaries.append(df_patient_seed)

    df_patient_seed = pd.concat(patient_seed_summaries, ignore_index=True)
    summaries = []

    for (intervention, level), df_level in df_patient_seed.groupby(['intervention', 'level'], sort=False):
        df_patient = df_level.groupby('patient_id')[['mean_difference', 'final_difference']].mean()
        df_seed = df_level.groupby('seed')[['mean_difference', 'final_difference']].mean()

        summaries.append({
            'intervention': intervention,
            'level': level,
            'mean_difference': df_patient['mean_difference'].mean(),
            'mean_difference_patient_sd': df_patient['mean_difference'].std(),
            'mean_difference_seed_sd': df_seed['mean_difference'].std(),
            'final_difference': df_patient['final_difference'].mean(),
            'final_difference_patient_sd': df_patient['final_difference'].std(),
            'final_difference_seed_sd': df_seed['final_difference'].std()
        })

    return pd.DataFrame(summaries)


def calculate_response_consistency(df):
    """
    Calculate directional and ranking consistency separately for each seed.
    """

    seed_summaries = []

    for intervention in ['insulin', 'carbohydrate']:
        df_intervention = df[df['intervention'] == intervention].copy()
        df_intervention['mean_cgm'] = df_intervention[PREDICTION_COLUMNS].mean(axis=1)

        df_wide = df_intervention.pivot(
            index=['seed', 'patient_id', 'state_index'], columns='value', values='mean_cgm')
        levels = sorted(df_wide.columns)

        for seed, df_seed in df_wide.groupby(level='seed'):
            reference_level = 0
            nonzero_levels = [level for level in levels if level > 0]
            direction_results = []

            for level in nonzero_levels:
                if intervention == 'insulin':
                    direction_results.append(df_seed[level].to_numpy() < df_seed[reference_level].to_numpy())
                else:
                    direction_results.append(df_seed[level].to_numpy() > df_seed[reference_level].to_numpy())

            direction_results = np.column_stack(direction_results)
            ranking_consistent = np.ones(len(df_seed), dtype=bool)

            for lower_level, higher_level in zip(levels[:-1], levels[1:]):
                if intervention == 'insulin':
                    ranking_consistent &= df_seed[lower_level].to_numpy() > df_seed[higher_level].to_numpy()
                else:
                    ranking_consistent &= df_seed[lower_level].to_numpy() < df_seed[higher_level].to_numpy()

            if intervention == 'insulin':
                expected_ordering = '0 U > 2 U > 5 U > 10 U'
            else:
                expected_ordering = '0 g < 10 g < 30 g < 50 g'

            seed_summaries.append({
                'seed': seed,
                'intervention': intervention,
                'expected_ordering': expected_ordering,
                'direction_count': direction_results.sum(),
                'num_direction_checks': direction_results.size,
                'direction_percent': 100 * direction_results.mean(),
                'ranking_count': ranking_consistent.sum(),
                'num_states': len(ranking_consistent),
                'ranking_percent': 100 * ranking_consistent.mean()
            })

    df_seed = pd.DataFrame(seed_summaries)
    summaries = []

    for (intervention, expected_ordering), group in df_seed.groupby(['intervention', 'expected_ordering'], sort=False):
        direction_checks_per_seed = group['num_direction_checks'].unique()
        ranking_states_per_seed = group['num_states'].unique()

        if len(direction_checks_per_seed) != 1:
            raise ValueError('Directional-check counts are not identical across seeds.')

        if len(ranking_states_per_seed) != 1:
            raise ValueError('Ranking-state counts are not identical across seeds.')

        summaries.append({
            'intervention': intervention,
            'expected_ordering': expected_ordering,
            'direction_percent': group['direction_percent'].mean(),
            'direction_percent_seed_sd': group['direction_percent'].std(),
            'num_direction_checks_per_seed': direction_checks_per_seed[0],
            'ranking_percent': group['ranking_percent'].mean(),
            'ranking_percent_seed_sd': group['ranking_percent'].std(),
            'num_ranking_states_per_seed': ranking_states_per_seed[0]
        })

    return pd.DataFrame(summaries)


def calculate_pairwise_consistency(df):
    """
    Calculate pairwise consistency separately for each seed.
    """

    seed_summaries = []

    for intervention in ['insulin', 'carbohydrate']:
        df_intervention = df[df['intervention'] == intervention].copy()
        df_intervention['mean_cgm'] = df_intervention[PREDICTION_COLUMNS].mean(axis=1)

        df_wide = df_intervention.pivot(
            index=['seed', 'patient_id', 'state_index'], columns='value', values='mean_cgm')
        levels = sorted(df_wide.columns)

        for seed, df_seed in df_wide.groupby(level='seed'):
            for i in range(len(levels)):
                for j in range(i + 1, len(levels)):
                    lower_level = levels[i]
                    higher_level = levels[j]

                    if intervention == 'insulin':
                        consistent = df_seed[lower_level].to_numpy() > df_seed[higher_level].to_numpy()
                        expected_ordering = f'{lower_level:g} U > {higher_level:g} U'
                    else:
                        consistent = df_seed[lower_level].to_numpy() < df_seed[higher_level].to_numpy()
                        expected_ordering = f'{lower_level:g} g < {higher_level:g} g'

                    seed_summaries.append({
                        'seed': seed,
                        'intervention': intervention,
                        'lower_level': lower_level,
                        'higher_level': higher_level,
                        'expected_ordering': expected_ordering,
                        'consistent_count': consistent.sum(),
                        'num_states': len(consistent),
                        'consistent_percent': 100 * consistent.mean()
                    })

    df_seed = pd.DataFrame(seed_summaries)
    summaries = []

    group_columns = ['intervention', 'lower_level', 'higher_level', 'expected_ordering']
    for keys, group in df_seed.groupby(group_columns, sort=False):
        intervention, lower_level, higher_level, expected_ordering = keys
        comparisons_per_seed = group['num_states'].unique()

        if len(comparisons_per_seed) != 1:
            raise ValueError('Pairwise-comparison counts are not identical across seeds.')

        summaries.append({
            'intervention': intervention,
            'lower_level': lower_level,
            'higher_level': higher_level,
            'expected_ordering': expected_ordering,
            'consistent_percent': group['consistent_percent'].mean(),
            'consistent_percent_seed_sd': group['consistent_percent'].std(),
            'num_comparisons_per_seed': comparisons_per_seed[0]
        })

    return pd.DataFrame(summaries)


def print_pairwise_consistency_table(df_summary):
    """
    Print pairwise consistency with SD across training seeds.
    """

    df_table = df_summary.copy()

    df_table['Intervention'] = df_table['intervention'].replace({'insulin': 'Insulin', 'carbohydrate': 'Carbohydrate'})

    df_table['Expected pairwise mean-CGM ordering'] = df_table['expected_ordering']
    df_table['Pairwise consistency'] = df_table.apply(
        lambda row: f'{row["consistent_percent"]:.1f}% ± '
                    f'{row["consistent_percent_seed_sd"]:.1f}% '
                    f'({row["num_comparisons_per_seed"]:.0f} comparisons/seed)', axis=1)

    df_table = df_table[['Intervention',
                         'Expected pairwise mean-CGM ordering',
                         'Pairwise consistency']]

    print('\nTable C. Pairwise counterfactual-level consistency')
    print(df_table.to_string(index=False))
    print('\nPercentages are mean ± SD across training seeds; comparisons are not pooled across seeds.')
    print('Pairwise comparisons use the mean predicted CGM over the 5–60-minute trajectory.')

    return df_table


def print_response_magnitude_table(df_summary):
    """
    Print response magnitude with patient SD and seed SD reported separately.
    """

    df_table = df_summary.copy()

    df_table['Intervention'] = df_table['intervention'].replace({'insulin': 'Insulin', 'carbohydrate': 'Carbohydrate'})
    df_table['Level'] = df_table.apply(
        lambda row: f'{row["level"]:g} U' if row['intervention'] == 'insulin'
        else f'{row["level"]:g} g', axis=1)

    df_table['Mean ΔG (mg/dL)'] = df_table.apply(
        lambda row: f'{row["mean_difference"]:.2f} ± '
                    f'{row["mean_difference_patient_sd"]:.2f}', axis=1)

    df_table['Mean ΔG seed SD'] = df_table['mean_difference_seed_sd'].map(lambda value: f'{value:.2f}')
    df_table['ΔG60 (mg/dL)'] = df_table.apply(
        lambda row: f'{row["final_difference"]:.2f} ± '
                    f'{row["final_difference_patient_sd"]:.2f}', axis=1)

    df_table['ΔG60 seed SD'] = df_table['final_difference_seed_sd'].map(lambda value: f'{value:.2f}')

    df_table = df_table[['Intervention',
                         'Level',
                         'Mean ΔG (mg/dL)',
                         'Mean ΔG seed SD',
                         'ΔG60 (mg/dL)',
                         'ΔG60 seed SD']]

    print('\nTable A. Counterfactual response magnitude')
    print(df_table.to_string(index=False))
    print('\nMain values are cohort mean ± SD across patient-level averages after averaging seeds.')
    print('Seed SD is calculated across seed-specific cohort means.')
    print('ΔG is calculated relative to the matched zero-action prediction.')

    return df_table


def print_response_consistency_table(df_summary):
    """
    Print directional and ranking consistency with SD across training seeds.
    """

    df_table = df_summary.copy()

    df_table['Intervention'] = df_table['intervention'].replace({'insulin': 'Insulin', 'carbohydrate': 'Carbohydrate'})

    df_table['Expected counterfactual-level ordering'] = df_table['expected_ordering']

    df_table['Directional consistency'] = df_table.apply(
        lambda row: f'{row["direction_percent"]:.1f}% ± '
                    f'{row["direction_percent_seed_sd"]:.1f}% '
                    f'({row["num_direction_checks_per_seed"]:.0f} checks/seed)', axis=1)

    df_table['Ranking consistency'] = df_table.apply(
        lambda row: f'{row["ranking_percent"]:.1f}% ± '
                    f'{row["ranking_percent_seed_sd"]:.1f}% '
                    f'({row["num_ranking_states_per_seed"]:.0f} states/seed)', axis=1)

    df_table = df_table[['Intervention',
                         'Expected counterfactual-level ordering',
                         'Directional consistency',
                         'Ranking consistency']]

    print('\nTable B. Counterfactual response consistency')
    print(df_table.to_string(index=False))
    print('\nPercentages are mean ± SD across training seeds; checks are not pooled across seeds.')
    print('Direction and ranking are calculated from the mean 5–60-minute response of each trajectory.')

    return df_table


def main():
    """
    Build the seed-averaged cohort figure and print the updated summary tables.
    """

    df = read_predictions()

    plot_cohort_trajectories(df, figure_file=MEAN_FIGURE_FILE)

    df_magnitude = calculate_response_magnitude(df)
    df_consistency = calculate_response_consistency(df)
    df_pairwise = calculate_pairwise_consistency(df)

    print_response_magnitude_table(df_magnitude)
    print_response_consistency_table(df_consistency)
    print_pairwise_consistency_table(df_pairwise)

    plt.close('all')


if __name__ == '__main__':
    # Run this once to train all patient-specific models for all seeds:
    run_experiment()

    # After glimmer_cf_seeds.csv has been generated, create the figure and tables:
    main()