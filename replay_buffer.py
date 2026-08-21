import numpy as np
import torch

from config import OfflineBufferConfig


class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, action, reward, next_state, done))
        else:
            self.storage[self.ptr] = (state, action, reward, next_state, done)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size, to_tensor=True, device="cpu"):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, r, s2, d = zip(*[self.storage[i] for i in ind])

        s = np.array(s)
        a = np.array(a)
        r = np.array(r).reshape(-1, 1)
        s2 = np.array(s2)
        d = np.array(d).reshape(-1, 1)
        not_d = 1.0 - d

        if to_tensor:
            s = torch.FloatTensor(s).to(device)
            a = torch.FloatTensor(a).to(device)
            r = torch.FloatTensor(r).to(device)
            s2 = torch.FloatTensor(s2).to(device)
            not_d = torch.FloatTensor(not_d).to(device)

        return s, a, r, s2, not_d

    def __len__(self):
        return len(self.storage)
    
    def fill_replay_buffer(self, env, seed):
        print("Filling shared offline replay buffer...")
        np.random.seed(seed)

        for i in range(OfflineBufferConfig.NUM_TRAIN_INIT_STATE):
            for _ in range(OfflineBufferConfig.MAX_EPISODES):
                state = env.reset(state_index=i, is_testing=False)
                episode_states, episode_actions, episode_rewards, episode_next_states = [], [], [], []

                for step in range(OfflineBufferConfig.MAX_STEPS_PER_EPISODE):
                    probs = np.random.dirichlet(np.ones(3))
                    action_type = np.argmax(probs)
                    carb_amount = np.random.uniform(*OfflineBufferConfig.CARB_RANGE)
                    insulin_amount = np.random.uniform(*OfflineBufferConfig.INSULIN_RANGE)
                    time_index = np.random.randint(0, 12)

                    action_vector = np.array(
                        [probs[0], probs[1], probs[2], carb_amount, insulin_amount, time_index],
                        dtype=np.float32
                    )

                    action_value = (
                        carb_amount if action_type == 1
                        else insulin_amount if action_type == 2
                        else 0.0
                    )
                    action = (action_type, action_value, time_index)

                    next_state, _, _, reward, _, _ = env.step(step, action)

                    episode_states.append(state)
                    episode_actions.append(action_vector)
                    episode_rewards.append(reward)
                    episode_next_states.append(next_state)

                    state = next_state

                episode_end_reward = env.compute_episode_reward()
                total_steps = len(episode_rewards)
                adjusted_rewards = [r + episode_end_reward / total_steps for r in episode_rewards]

                for s, a, r, s_next in zip(
                    episode_states,
                    episode_actions,
                    adjusted_rewards,
                    episode_next_states
                ):
                    self.add(s, a, r, s_next, False)

        print(f"Shared offline replay buffer filled with {len(self)} transitions.")
