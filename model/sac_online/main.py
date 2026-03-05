import os
import pickle
import sys

import numpy as np
import torch

from config import Action
from config import SACConfig, EnvConfig, DataConfig
from environment import Environment
from model.sac.sac_agent import SACAgent
from model.td3_bc.replay_buffer import ReplayBuffer
from utils import (plot_cgm_reward_action, cal_time_in_range, cal_time_below_range, cal_time_above_range, cal_coefficient_of_variation,
                   plot_tir_tbr_tar, plot_eat_action_distribution, plot_insulin_action_distribution, extract_behavior_features_from_actions,
                   extract_patient_behavior_features, plot_behavior_radar, set_seed)


def train_sac_online(env, env_eval, agent, buffer, folder_path):
    print("Starting ONLINE SAC training...")
    eval_return = []

    global_step = 0

    for epoch in range(SACConfig.MAX_EPOCHS):
        for i in range(SACConfig.NUM_TRAIN_INIT_STATE):
            state = env.reset(state_index=i, is_testing=False)

            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []

            for step in range(SACConfig.MAX_STEPS_PER_EPISODE):
                # --- select action from current policy ---
                action_type, value, time_index = agent.select_action(state)
                action = (action_type, value, time_index)

                next_state, _, _, reward, done, _ = env.step(step, action)

                # --- build action vector (same format as offline SAC) ---
                action_vec = np.zeros(6, dtype=np.float32)
                action_vec[action_type] = 1.0

                if action_type == Action.EAT:
                    action_vec[3] = value
                elif action_type == Action.INJECT:
                    action_vec[4] = value

                action_vec[5] = time_index

                episode_states.append(state)
                episode_actions.append(action_vec)
                episode_rewards.append(reward)
                episode_next_states.append(next_state)

                state = next_state
                global_step += 1

                # --- train SAC after warmup ---
                if len(buffer) >= SACConfig.BATCH_SIZE:
                    agent.train(buffer, SACConfig.BATCH_SIZE)

            # ---- episode-end reward shaping (same logic as before) ----
            episode_end_reward = env.compute_episode_reward()
            n_steps = len(episode_rewards)

            if n_steps > 0:
                bonus = episode_end_reward / n_steps
                episode_rewards = [r + bonus for r in episode_rewards]

            # ---- add episode transitions to replay buffer ----
            for s, a, r, s_next in zip(
                episode_states,
                episode_actions,
                episode_rewards,
                episode_next_states
            ):
                buffer.add(s, a, r, s_next, False)

        # ---- evaluation ----
        avg_return = evaluate_agent(env_eval, agent)
        eval_return.append((global_step, avg_return))
        print(f"[Eval] Step {global_step} | Avg return: {avg_return:.2f}")

    # ---- save learning curve ----
    save_path = os.path.join(folder_path, "learning_curve.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(eval_return, f)

    print("ONLINE SAC training finished and learning curve saved.")


def evaluate_agent(env, agent):
    returns = []

    for i in range(SACConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)
        total_reward = 0.0

        for step in range(SACConfig.TESTING_STEPS):
            action_type, value, time_index = agent.select_action(state)

            action = (action_type, value, time_index)
            state, _, _, reward, _, _ = env.step(step, action)
            total_reward += reward

        total_reward += env.compute_episode_reward()
        returns.append(total_reward)

    return np.mean(returns)


def test_sac(env, agent, folder_path):
    print("Evaluating policy...")

    log_path = os.path.join(folder_path, "eval_results.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)

    test_rewards, test_cgms, test_actions, test_time_window, test_behavioral_features = [], [], [], [], []
    test_tir, test_tar, test_tbr, test_cv = [], [], [], []

    y_history = env.simulator.data.y_history
    x_history = env.simulator.data.X_history
    patient_behavioral_features = extract_patient_behavior_features(x_history)

    for i in range(SACConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)

        total_reward = 0
        rewards, predicted_cgms, actions, time_window = [], [], [], []

        print(f"\n--------------------- Test {i + 1} ---------------------")
        for step in range(SACConfig.TESTING_STEPS):
            action_type, value, time_index = agent.select_action(state)
            action = (action_type, value, time_index)

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
        cv = cal_coefficient_of_variation(predicted_cgms)

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
    folder_path = f'./model/sac_online/tests/{dataset_name}/{dataset_name}_patient_{patient_id}/seed_{seed}/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    env = Environment(dataset_name=dataset_name, patient_id=patient_id)
    env_eval = Environment(dataset_name=dataset_name, patient_id=patient_id)

    max_action = np.array([1.0, 1.0, 1.0, SACConfig.CARB_RANGE[1], SACConfig.INSULIN_RANGE[1], 11.0])

    agent = SACAgent(state_dim=EnvConfig.STATE_DIM, action_dim=len(max_action), max_action=max_action, device=device)
    buffer = ReplayBuffer()

    train_sac_online(env, env_eval, agent, buffer, folder_path)
    test_sac(env, agent, folder_path)


if __name__ == "__main__":
    set_seed(DataConfig.SEEDS[0])
    main(dataset_name=DataConfig.DATASET, patient_id=str(DataConfig.PATIENT_ID), seed=DataConfig.SEEDS[0])