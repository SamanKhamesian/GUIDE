import os
import sys

import numpy as np
import torch

from config import Action, CQLConfig, DataConfig, EnvConfig
from environment import Environment
from model.cql_bc.cql_agent import CQL
from replay_buffer import ReplayBuffer
from utils import set_seed, cal_time_in_range, cal_time_below_range, cal_time_above_range, cal_coefficient_of_variation


ADHERENCE_CONDITIONS = {
    "adherence_100": 1.00,
    "adherence_75": 0.75,
    "adherence_50": 0.50,
}


def train_cql(agent, buffer):
    print("Starting CQL-BC training...")

    for step in range(CQLConfig.TRAINING_STEPS):
        agent.train(buffer, batch_size=CQLConfig.BATCH_SIZE)

        if step % 200 == 0:
            print(f"[CQL-BC Train] step {step}/{CQLConfig.TRAINING_STEPS}")

    print("\nCQL-BC training finished!")


def apply_adherence(action, adherence_probability, adherence_rng):
    action_type, action_value, time_index = action

    if action_type == Action.NOTHING:
        return action, False, False

    followed = adherence_rng.random() < adherence_probability

    if followed:
        return action, True, True

    return (Action.NOTHING, 0.0, time_index), True, False


def test_cql(env, agent, action_low, action_high, folder_path, condition_name, adherence_probability, seed):
    print(f"Evaluating {condition_name}...")

    os.makedirs(folder_path, exist_ok=True)
    log_path = f"{folder_path}/eval_results.txt"

    if os.path.isfile(log_path):
        os.remove(log_path)

    adherence_rng = np.random.default_rng(seed + 10_000)

    test_tir, test_tbr, test_tar, test_cv = [], [], [], []
    total_recommendations = 0
    total_followed = 0

    for i in range(CQLConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)
        predicted_cgms = []

        print(f"\n--------------------- Test {i + 1} ---------------------")

        for step in range(CQLConfig.TESTING_STEPS):
            raw_action = agent.select_action(state)
            raw_action = np.clip(raw_action, action_low, action_high)

            scores = raw_action[:3]
            action_type = int(np.argmax(scores))
            carb_amount = raw_action[3]
            insulin_amount = raw_action[4]
            time_index = int(np.clip(np.round(raw_action[5]), 0, 11))

            action_value = carb_amount if action_type == Action.EAT else insulin_amount if action_type == Action.INJECT else 0.0
            recommended_action = (action_type, action_value, time_index)
            executed_action, is_recommendation, followed = apply_adherence(
                recommended_action,
                adherence_probability,
                adherence_rng,
            )

            if is_recommendation:
                total_recommendations += 1
                total_followed += int(followed)

            state, predicted_cgm, _, _, _, _ = env.step(step, executed_action)
            predicted_cgms.extend(predicted_cgm)

        tir = cal_time_in_range(predicted_cgms)
        tbr = cal_time_below_range(predicted_cgms)
        tar = cal_time_above_range(predicted_cgms)
        cv = cal_coefficient_of_variation(predicted_cgms)

        test_tir.append(tir)
        test_tbr.append(tbr)
        test_tar.append(tar)
        test_cv.append(cv)

        print(f"\n-------------- Results for Test {i + 1} ---------------\n")
        print(f"Time-in-Range           : {tir:.2f}%")
        print(f"Time-below-Range        : {tbr:.2f}%")
        print(f"Time-above-Range        : {tar:.2f}%")
        print(f"Coefficient of Variation: {cv:.2f}%")

        with open(log_path, "a") as f:
            f.write(f"\n--------------------- Results for Test {i + 1} ----------------------\n")
            f.write(f"Time-in-Range (TIR)          : {tir:.2f}%\n")
            f.write(f"Time-below-Range (TBR)       : {tbr:.2f}%\n")
            f.write(f"Time-above-Range (TAR)       : {tar:.2f}%\n")
            f.write(f"Coefficient of Variation (CV): {cv:.2f}%\n")

    realized_adherence = 100.0 if total_recommendations == 0 else 100.0 * total_followed / total_recommendations
    evaluate_performance(
        test_tir,
        test_tbr,
        test_tar,
        test_cv,
        log_path,
        adherence_probability,
        realized_adherence,
        total_recommendations,
    )


def evaluate_performance(test_tir, test_tbr, test_tar, test_cv, log_path, adherence_probability, realized_adherence, total_recommendations):
    print("\n----------------- Final Results ------------------")

    avg_tir = np.mean(test_tir)
    std_tir = np.std(test_tir)
    med_tir = np.median(test_tir)

    avg_tbr = np.mean(test_tbr)
    std_tbr = np.std(test_tbr)
    med_tbr = np.median(test_tbr)

    avg_tar = np.mean(test_tar)
    std_tar = np.std(test_tar)
    med_tar = np.median(test_tar)

    avg_cv = np.mean(test_cv)
    std_cv = np.std(test_cv)
    med_cv = np.median(test_cv)

    print(f"\nTarget adherence                    : {100 * adherence_probability:.0f}%")
    print(f"Realized adherence                  : {realized_adherence:.2f}%")
    print(f"Total EAT/INJECT recommendations    : {total_recommendations}")
    print(f"Average Time-in-Range (TIR)          : {avg_tir:.2f}% (± {std_tir:.2f}%), Median: {med_tir:.2f}%")
    print(f"Average Time-below-Range (TBR)       : {avg_tbr:.2f}% (± {std_tbr:.2f}%), Median: {med_tbr:.2f}%")
    print(f"Average Time-above-Range (TAR)       : {avg_tar:.2f}% (± {std_tar:.2f}%), Median: {med_tar:.2f}%")
    print(f"Average Coefficient of Variation (CV): {avg_cv:.2f}% (± {std_cv:.2f}%), Median: {med_cv:.2f}%")

    with open(log_path, "a") as f:
        f.write("\n---------------- Adherence Summary ----------------\n")
        f.write(f"Target adherence                    : {100 * adherence_probability:.0f}%\n")
        f.write(f"Realized adherence                  : {realized_adherence:.2f}%\n")
        f.write(f"Total EAT/INJECT recommendations    : {total_recommendations}\n")
        f.write("\n----------- Summary Time-in-Range Stats Across Tests -----------\n")
        f.write(f"Average Time-in-Range (TIR)          : {avg_tir:.2f}% (± {std_tir:.2f}%), Median: {med_tir:.2f}%\n")
        f.write(f"Average Time-below-Range (TBR)       : {avg_tbr:.2f}% (± {std_tbr:.2f}%), Median: {med_tbr:.2f}%\n")
        f.write(f"Average Time-above-Range (TAR)       : {avg_tar:.2f}% (± {std_tar:.2f}%), Median: {med_tar:.2f}%\n")
        f.write(f"Average Coefficient of Variation (CV): {avg_cv:.2f}% (± {std_cv:.2f}%), Median: {med_cv:.2f}%\n")


def main(dataset_name, patient_id, seed):
    device = torch.device("cpu")
    env = Environment(dataset_name=dataset_name, patient_id=patient_id)

    action_low = np.array([0, 0, 0, CQLConfig.CARB_RANGE[0], CQLConfig.INSULIN_RANGE[0], 0], dtype=np.float32)
    action_high = np.array([1, 1, 1, CQLConfig.CARB_RANGE[1], CQLConfig.INSULIN_RANGE[1], 11], dtype=np.float32)

    agent = CQL(state_dim=EnvConfig.STATE_DIM, action_dim=len(action_high), max_action=action_high, device=device)

    buffer = ReplayBuffer()
    buffer.fill_replay_buffer(env)

    train_cql(agent, buffer)

    evaluation_random_state = np.random.get_state()

    for condition_name, adherence_probability in ADHERENCE_CONDITIONS.items():
        np.random.set_state(evaluation_random_state)

        folder_path = (f"./adherence_study/tests/{condition_name}/{dataset_name}/"
                       f"{dataset_name}_patient_{patient_id}/seed_{seed}/")

        test_cql(env, agent, action_low, action_high, folder_path, condition_name, adherence_probability, seed)


if __name__ == "__main__":
    patient_id = sys.argv[1]
    seed = int(sys.argv[2])

    set_seed(seed)
    main(dataset_name=DataConfig.DATASET, patient_id=patient_id, seed=seed)
