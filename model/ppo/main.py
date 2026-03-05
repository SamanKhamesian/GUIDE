import gc
import os
import pickle
import sys

import numpy as np
import torch

from config import Action, PPOConfig, DataConfig, EnvConfig
from environment import Environment
from model.ppo.buffer import PPOBuffer
from model.ppo.ppo_agent import PPOAgent
from utils import (plot_cgm_reward_action, set_seed, cal_time_in_range, cal_time_above_range, cal_time_below_range, plot_tir_tbr_tar,
                   plot_eat_action_distribution, plot_insulin_action_distribution, extract_patient_behavior_features,
                   extract_behavior_features_from_actions, cal_coefficient_of_variation, plot_behavior_radar)


def train_ppo(env, env_eval, agent, buffer, folder_path):
    print("Starting PPO training...")
    eval_returns = []

    for epoch in range(PPOConfig.MAX_EPOCHS):
        for i in range(PPOConfig.NUM_TRAIN_INIT_STATE):
            state = env.reset(state_index=i, is_testing=False)
            print("\n-------------------- Epoch {}, Step {} --------------------".format(epoch + 1, i + 1))
            start_idx = len(buffer.rewards)

            for step in range(PPOConfig.MAX_STEPS_PER_EPISODE):

                with torch.no_grad():
                    action_type, carb_amount, insulin_amount, time_index, log_prob = agent.select_action(state)
                    value_estimate = agent.evaluate(state)

                if action_type == Action.EAT:
                    value = carb_amount
                elif action_type == Action.INJECT:
                    value = insulin_amount
                else:
                    value = 0.0

                action = (action_type, value, time_index)

                next_state, _, _, reward, done, _ = env.step(step, action)

                buf_state = np.asarray(state, dtype=np.float32)
                buf_action = [int(action_type), float(carb_amount), float(insulin_amount), int(time_index)]
                buf_reward = float(reward)
                buf_value = float(value_estimate)
                buf_logp = float(log_prob)

                buffer.store(buf_state, buf_action, buf_reward, buf_value, buf_logp, bool(done))

                state = next_state

            # Add episode-end reward to only this episode’s steps
            ep_end_reward = env.compute_episode_reward()
            end_idx = len(buffer.rewards)
            steps_this_ep = end_idx - start_idx
            if steps_this_ep > 0:
                add_per_step = ep_end_reward / steps_this_ep
                for k in range(start_idx, end_idx):
                    buffer.rewards[k] += add_per_step

        # Train after all init states, then reset buffer and free memory
        agent.train(buffer)
        buffer.reset()

        # ---- Evaluation ----
        avg_return = evaluate_agent(env_eval, agent)
        eval_returns.append((epoch, avg_return))
        print(f"[Eval] Epoch {epoch}/{PPOConfig.MAX_EPOCHS} | Avg return: {avg_return:.2f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # save learning curve
    save_path = os.path.join(folder_path, "learning_curve.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(eval_returns, f)

    print("Training finished and learning curve saved.")


def evaluate_agent(env, agent):
    returns = []

    for i in range(PPOConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)
        total_reward = 0.0

        for step in range(PPOConfig.TESTING_STEPS):
            with torch.no_grad():
                action_type, carb_amount, insulin_amount, time_index, _ = agent.select_action(state)

            value = (
                carb_amount if action_type == Action.EAT
                else insulin_amount if action_type == Action.INJECT
                else 0.0
            )

            action = (action_type, value, time_index)
            state, _, _, reward, _, _ = env.step(step, action)
            total_reward += reward

        total_reward += env.compute_episode_reward()
        returns.append(total_reward)

    return float(np.mean(returns))


def test_ppo(env, agent, folder_path):
    print("Evaluating policy...")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    log_path = os.path.join(folder_path, "eval_results.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)

    test_rewards, test_cgms, test_actions, test_time_window, test_behavioral_features = [], [], [], [], []
    test_tir, test_tar, test_tbr, test_cv = [], [], [], []

    y_history = env.simulator.data.y_history
    x_history = env.simulator.data.X_history
    patient_behavioral_features = extract_patient_behavior_features(x_history)

    for i in range(PPOConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)

        total_reward = 0
        rewards, predicted_cgms, actions, time_window = [], [], [], []

        print(f"\n--------------------- Test {i + 1} ---------------------")
        for step in range(PPOConfig.TESTING_STEPS):
            action_type, carb_amount, insulin_amount, time_index, _ = agent.select_action(state)

            mapped_type = action_type
            mapped_value = carb_amount if mapped_type == Action.EAT else (insulin_amount if mapped_type == Action.INJECT else 0.0)
            mapped_time = time_index

            action = (mapped_type, mapped_value, mapped_time)
            state, predicted_cgm, ts, reward, done, _ = env.step(step, action)

            total_reward += reward
            rewards.append(reward)
            actions.append(action)
            predicted_cgms.extend(predicted_cgm)
            time_window.extend(ts)

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
    folder_path = f"./ppo_model/tests/{dataset_name}/{dataset_name}_patient_{patient_id}/seed_{seed}/"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    env = Environment(dataset_name=dataset_name, patient_id=patient_id)
    env_eval = Environment(dataset_name=dataset_name, patient_id=patient_id)

    agent = PPOAgent(state_dim=EnvConfig.STATE_DIM, device=device)
    buffer = PPOBuffer(state_dim=EnvConfig.STATE_DIM, device=device)

    train_ppo(env, env_eval, agent, buffer, folder_path)
    test_ppo(env, agent, folder_path)


if __name__ == "__main__":
    set_seed(DataConfig.SEEDS[0])
    main(dataset_name=DataConfig.DATASET, patient_id=str(DataConfig.PATIENT_ID), seed=DataConfig.SEEDS[0])
