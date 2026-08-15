import os
import re

import numpy as np
from utils import cal_wilcoxon_matrix


PATIENTS = list(range(1, 26))
SEEDS = [42, 43, 44, 45, 46]


# Change the addresses if needed.
# use_seed_folder=False is used for Random because its results are saved
# directly inside each patient folder.
ALGORITHMS = {
    "TD3-BC": {
        "path": "./model/td3_bc/tests/final/azt1d",
        "filename": "eval_results.txt",
        "use_seed_folder": True,
    },
    "CQL-BC": {
        "path": "./model/cql_bc/tests/azt1d",
        "filename": "eval_results.txt",
        "use_seed_folder": True,
    },
    "PPO": {
        "path": "./model/ppo/tests/azt1d",
        "filename": "eval_results.txt",
        "use_seed_folder": True,
    },
    "SAC-Offline": {
        "path": "./model/sac/tests/azt1d",
        "filename": "eval_results.txt",
        "use_seed_folder": True,
    },
    "SAC-Online": {
        "path": "./model/sac_online/tests/azt1d",
        "filename": "eval_results.txt",
        "use_seed_folder": True,
    },
    "Random": {
        "path": "./model/random/tests/azt1d",
        "filename": "eval_results_random.txt",
        "use_seed_folder": False,
    },
}


METRIC_PATTERNS = {
    "tir": r"Average Time-in-Range \(TIR\)\s*:\s*([\d.]+)%",
    "tar": r"Average Time-above-Range \(TAR\)\s*:\s*([\d.]+)%",
    "tbr": r"Average Time-below-Range \(TBR\)\s*:\s*([\d.]+)%",
    "cv": r"Average Coefficient of Variation \(CV\)\s*:\s*([\d.]+)%",
}


def read_metrics(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    results = {}

    for metric, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, content)

        if match is None:
            raise ValueError(f"Could not find {metric.upper()} in {file_path}")

        results[metric] = float(match.group(1))

    return results


def collect_algorithm_results(base_path, filename, use_seed_folder):
    results = {
        "tir": [],
        "tar": [],
        "tbr": [],
        "cv": [],
    }

    for patient_id in PATIENTS:
        patient_results = {
            "tir": [],
            "tar": [],
            "tbr": [],
            "cv": [],
        }

        patient_folder = os.path.join(base_path, f"azt1d_patient_{patient_id}")

        if use_seed_folder:
            file_paths = [
                os.path.join(patient_folder, f"seed_{seed}", filename)
                for seed in SEEDS
            ]
        else:
            file_paths = [os.path.join(patient_folder, filename)]

        for file_path in file_paths:
            seed_results = read_metrics(file_path)

            for metric in patient_results:
                patient_results[metric].append(seed_results[metric])

        for metric in results:
            patient_average = np.mean(patient_results[metric])
            results[metric].append(round(float(patient_average), 2))

    return results


def print_df(df):
    print("df = {")

    algorithms = list(df.keys())

    for algorithm_index, algorithm in enumerate(algorithms):
        print(f'    "{algorithm}": {{')

        metrics = list(df[algorithm].keys())

        for metric_index, metric in enumerate(metrics):
            comma = "," if metric_index < len(metrics) - 1 else ""
            print(f'        "{metric}": {df[algorithm][metric]}{comma}')

        algorithm_comma = "," if algorithm_index < len(algorithms) - 1 else ""
        print(f"    }}{algorithm_comma}")

    print("}")


def print_final_results(df):
    print("\nFinal Results (Mean +/- SD across patients):")

    for algorithm, results in df.items():
        print(f"\n{algorithm}")

        for metric, values in results.items():
            average = np.mean(values)
            std = np.std(values, ddof=1)
            print(f"{metric.upper()}: {average:.2f} +/- {std:.2f}")


if __name__ == "__main__":
    df = {}

    for algorithm, config in ALGORITHMS.items():
        df[algorithm] = collect_algorithm_results(
            base_path=config["path"],
            filename=config["filename"],
            use_seed_folder=config["use_seed_folder"],
        )

    print_df(df)
    print_final_results(df)

    cal_wilcoxon_matrix(df, ["TD3-BC", "CQL-BC", "PPO", "SAC-Offline", "SAC-Online", "Random"], "tir", correction="holm")
