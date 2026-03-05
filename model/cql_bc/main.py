import os
import pickle
import sys

import numpy as np
import torch

from config import CQLConfig, DataConfig, EnvConfig
from environment import Environment
from model.cql_bc.cql_agent import CQL
from model.td3_bc.replay_buffer import ReplayBuffer
from utils import (plot_cgm_reward_action, set_seed, cal_time_in_range, cal_time_below_range, cal_time_above_range, cal_coefficient_of_variation,
                   plot_tir_tbr_tar, plot_eat_action_distribution, plot_insulin_action_distribution, extract_behavior_features_from_actions,
                   extract_patient_behavior_features, plot_behavior_radar)


def fill_replay_buffer(env, buffer):
    for i in range(CQLConfig.NUM_TRAIN_INIT_STATE):
        for _ in range(CQLConfig.MAX_EPISODES):
            state = env.reset(state_index=i, is_testing=False)

            states, actions, rewards, next_states = [], [], [], []

            for step in range(CQLConfig.MAX_STEPS_PER_EPISODE):
                scores = np.random.randn(3)
                action_type = np.argmax(scores)
                carb = np.random.uniform(*CQLConfig.CARB_RANGE)
                insulin = np.random.uniform(*CQLConfig.INSULIN_RANGE)
                time_idx = np.random.randint(0, 12)

                action_vector = np.array([scores[0], scores[1], scores[2], carb, insulin, time_idx], dtype=np.float32)
                env_action = (action_type, carb if action_type == 1 else insulin if action_type == 2 else 0.0, time_idx)

                next_state, _, _, reward, _, _ = env.step(step, env_action)

                states.append(state)
                actions.append(action_vector)
                rewards.append(reward)
                next_states.append(next_state)

                state = next_state

            episode_reward = env.compute_episode_reward()
            rewards = [r + episode_reward / len(rewards) for r in rewards]

            for s, a, r, s_next in zip(states, actions, rewards, next_states):
                buffer.add(s, a, r, s_next, False)

    print("Replay buffer is ready.")


def train_cql(env, agent, buffer, action_low, action_high, folder_path):
    print("Starting CQL training...")
    eval_return = []

    for step in range(CQLConfig.TRAINING_STEPS):
        agent.train(buffer, batch_size=CQLConfig.BATCH_SIZE)

        if step % 200 == 0:
            print(f"[CQL Train] step {step}/{CQLConfig.TRAINING_STEPS}")
            avg_return = evaluate_agent(env, agent, action_low, action_high)
            eval_return.append((step, avg_return))

    save_path = os.path.join(folder_path, "learning_curve.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(eval_return, f)

    print(f"\nCQL training finished and learning curve saved!")


def evaluate_agent(env, agent, action_low, action_high):
    returns = []

    for i in range(CQLConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)
        total_reward = 0

        for step in range(CQLConfig.TESTING_STEPS):
            raw_action = agent.select_action(state)
            raw_action = np.clip(raw_action, action_low, action_high)

            scores = raw_action[:3]
            action_type = int(np.argmax(scores))
            carb = raw_action[3]
            insulin = raw_action[4]
            time_idx = int(np.clip(np.round(raw_action[5]), 0, 11))

            value = carb if action_type == 1 else insulin if action_type == 2 else 0.0
            action = (action_type, value, time_idx)

            state, _, _, reward, _, _ = env.step(step, action)
            total_reward += reward

        total_reward += env.compute_episode_reward()
        returns.append(total_reward)

    return np.mean(returns)


def test_cql(env, agent, action_low, action_high, folder_path):
    print("Evaluating policy...")

    log_path = os.path.join(folder_path, "eval_results.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)

    test_rewards, test_cgms, test_actions, test_time_window, test_behavioral_features = [], [], [], [], []
    test_tir, test_tar, test_tbr, test_cv = [], [], [], []

    y_history = env.simulator.data.y_history
    x_history = env.simulator.data.X_history
    patient_behavioral_features = extract_patient_behavior_features(x_history)

    for i in range(CQLConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)

        total_reward = 0
        rewards, predicted_cgms, actions, time_window = [], [], [], []

        print(f"\n--------------------- Test {i + 1} ---------------------")
        for step in range(CQLConfig.TESTING_STEPS):
            raw_action = agent.select_action(state)
            raw_action = np.clip(raw_action, action_low, action_high)

            probs = raw_action[:3]
            mapped_type = int(np.argmax(probs))
            carb_amt = raw_action[3]
            insulin_amt = raw_action[4]

            mapped_time = int(np.clip(np.round(raw_action[5]), 0, 11))
            mapped_value = (carb_amt if mapped_type == 1 else insulin_amt if mapped_type == 2 else 0.0)

            action = (mapped_type, mapped_value, mapped_time)
            state, predicted_cgm, time_series, reward, _, _ = env.step(step, action)

            total_reward += reward
            rewards.append(reward)
            actions.append(action)
            predicted_cgms.extend(predicted_cgm)
            time_window.extend(time_series)

        main_meal_actions = env.main_meal_action_log.copy()
        episode_end_reward = env.compute_episode_reward()
        total_reward += episode_end_reward

        agent_features = extract_behavior_features_from_actions(actions, main_meal_actions)
        test_behavioral_features.append(agent_features)

        test_rewards.append(total_reward)
        test_cgms.append(predicted_cgms)
        test_actions.append(actions)
        test_time_window.append(time_window)

        print(f"\n-------------- Results for Test {i + 1} ---------------\n")
        print(f"Episode-end reward      : {episode_end_reward:.2f}")
        print(f"Evaluation reward       : {total_reward - episode_end_reward:.2f}")
        print(f"Total Evaluation reward : {total_reward:.2f}")

        tir = cal_time_in_range(predicted_cgms)
        tar = cal_time_above_range(predicted_cgms)
        tbr = cal_time_below_range(predicted_cgms)
        cv  = cal_coefficient_of_variation(predicted_cgms)

        test_tir.append(tir)
        test_tar.append(tar)
        test_tbr.append(tbr)
        test_cv.append(cv)

        print(f"\nTime-in-Range           : {tir:.2f}%")
        print(f"Time-above-Range        : {tar:.2f}%")
        print(f"Time-below-Range        : {tbr:.2f}%")
        print(f"Coefficient of Variation: {cv:.2f}%")

        plot_cgm_reward_action(cgm_sequence=predicted_cgms,
                               hour_series=time_window,
                               reward_list=rewards,
                               action_list=actions,
                               test_index=i + 1,
                               main_meal_actions=main_meal_actions,
                               save_path_prefix=folder_path)

        with open(log_path, "a") as f:
            f.write(f"\n--------------------- Results for Test {i + 1} ----------------------\n")
            f.write(f"\nEpisode-end reward      : {episode_end_reward:.2f}\n")
            f.write(f"Evaluation reward       : {total_reward - episode_end_reward:.2f}\n")
            f.write(f"Total Evaluation reward : {total_reward:.2f}\n")
            f.write(f"\nTime-in-Range           : {tir:.2f}%\n")
            f.write(f"Time-above-Range        : {tar:.2f}%\n")
            f.write(f"Time-below-Range        : {tbr:.2f}%\n")
            f.write(f"Coefficient of Variation: {cv:.2f}%\n")

    evaluate_performance(test_actions, test_time_window, y_history, test_tir, test_tar, test_tbr, test_cv, log_path, folder_path)

    if test_behavioral_features:
        avg_features = {}
        keys = test_behavioral_features[0].keys()
        for key in keys:
            values = [f[key] for f in test_behavioral_features]
            avg_features[key] = round(np.mean(values), 2)

        plot_behavior_radar(patient_behavioral_features, avg_features, save_path=folder_path)


def evaluate_performance(test_actions, test_time_window, y_history, test_tir, test_tar, test_tbr, test_cv, log_path, folder_path):
    print(f"\n----------------- Final Results ------------------")

    # TIR
    avg_tir = np.mean(test_tir)
    std_tir = np.std(test_tir)
    med_tir = np.median(test_tir)

    # TAR
    avg_tar = np.mean(test_tar)
    std_tar = np.std(test_tar)
    med_tar = np.median(test_tar)

    # TBR
    avg_tbr = np.mean(test_tbr)
    std_tbr = np.std(test_tbr)
    med_tbr = np.median(test_tbr)

    # CV
    avg_cv = np.mean(test_cv)
    std_cv = np.std(test_cv)
    med_cv = np.median(test_cv)

    plot_tir_tbr_tar(test_tir, test_tar, test_tbr, save_path=f'{folder_path}/all_tests.png')
    plot_eat_action_distribution(test_actions, test_time_window, save_path=folder_path)
    plot_insulin_action_distribution(test_actions, test_time_window, save_path=folder_path)

    print(f"\nAverage Time-in-Range (TIR)          : {avg_tir:.2f}% (± {std_tir:.2f}%), Median: {med_tir:.2f}%")
    print(f"Average Time-above-Range (TAR)       : {avg_tar:.2f}% (± {std_tar:.2f}%), Median: {med_tar:.2f}%")
    print(f"Average Time-below-Range (TBR)       : {avg_tbr:.2f}% (± {std_tbr:.2f}%), Median: {med_tbr:.2f}%")
    print(f"Average Coefficient of Variation (CV): {avg_cv:.2f}% (± {std_cv:.2f}%), Median: {med_cv:.2f}%")

    print(f"\nHistory Time-in-Range (TIR)          : {cal_time_in_range(y_history):.2f}%")
    print(f"History Time-above-Range (TAR)       : {cal_time_above_range(y_history):.2f}%")
    print(f"History Time-below-Range (TBR)       : {cal_time_below_range(y_history):.2f}%")
    print(f"History Coefficient of Variation (CV): {cal_coefficient_of_variation(y_history):.2f}%")

    with open(log_path, "a") as f:
        # Summary statistics block
        f.write("\n----------- Summary Time-in-Range Stats Across Tests -----------\n")
        f.write(f"Average Time-in-Range (TIR)          : {avg_tir:.2f}% (± {std_tir:.2f}%), Median: {med_tir:.2f}%\n")
        f.write(f"Average Time-above-Range (TAR)       : {avg_tar:.2f}% (± {std_tar:.2f}%), Median: {med_tar:.2f}%\n")
        f.write(f"Average Time-below-Range (TBR)       : {avg_tbr:.2f}% (± {std_tbr:.2f}%), Median: {med_tbr:.2f}%\n")
        f.write(f"Average Coefficient of Variation (CV): {avg_cv:.2f}% (± {std_cv:.2f}%), Median: {med_cv:.2f}%\n")

        # History block
        f.write("\n---------------- Historical Time-in-Range Stats ----------------\n")
        f.write(f"History Time-in-Range (TIR)          : {cal_time_in_range(y_history):.2f}%\n")
        f.write(f"History Time-above-Range (TAR)       : {cal_time_above_range(y_history):.2f}%\n")
        f.write(f"History Time-below-Range (TBR)       : {cal_time_below_range(y_history):.2f}%\n")
        f.write(f"History Coefficient of Variation (CV): {cal_coefficient_of_variation(y_history):.2f}%")


def main(dataset_name, patient_id, seed):
    device = torch.device("cpu")
    folder_path = f'./model/cql_bc/tests/final/{dataset_name}/{dataset_name}_patient_{patient_id}/seed_{seed}/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    env = Environment(dataset_name=dataset_name, patient_id=patient_id)
    env_eval = Environment(dataset_name=dataset_name, patient_id=patient_id)

    action_low = np.array([0, 0, 0, CQLConfig.CARB_RANGE[0], CQLConfig.INSULIN_RANGE[0], 0], dtype=np.float32)
    action_high = np.array([1, 1, 1, CQLConfig.CARB_RANGE[1], CQLConfig.INSULIN_RANGE[1], 11], dtype=np.float32)
    agent = CQL(state_dim=EnvConfig.STATE_DIM, action_dim=len(action_high), max_action=action_high, device=device)
    buffer = ReplayBuffer()

    fill_replay_buffer(env, buffer)
    train_cql(env_eval, agent, buffer, action_low, action_high, folder_path)
    test_cql(env, agent, action_low, action_high, folder_path)

if __name__ == "__main__":
    set_seed(DataConfig.SEEDS[0])
    main(dataset_name=DataConfig.DATASET, patient_id=str(DataConfig.PATIENT_ID), seed=DataConfig.SEEDS[0])
