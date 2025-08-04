import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from config import PPOConfig


class PPOActor(nn.Module):
    def __init__(self, state_dim, n_action_type, n_carb, n_insulin, n_time):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, PPOConfig.HIDDEN_SIZE)
        self.fc2 = nn.Linear(PPOConfig.HIDDEN_SIZE, PPOConfig.HIDDEN_SIZE)

        self.logits_type = nn.Linear(PPOConfig.HIDDEN_SIZE, n_action_type)
        self.logits_carb = nn.Linear(PPOConfig.HIDDEN_SIZE, n_carb)
        self.logits_insulin = nn.Linear(PPOConfig.HIDDEN_SIZE, n_insulin)
        self.logits_time = nn.Linear(PPOConfig.HIDDEN_SIZE, n_time)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits_type(x), self.logits_carb(x), self.logits_insulin(x), self.logits_time(x)

    def get_action(self, state):
        logits_type, logits_carb, logits_insulin, logits_time = self.forward(state)

        dist_type = Categorical(logits=logits_type)
        dist_carb = Categorical(logits=logits_carb)
        dist_insulin = Categorical(logits=logits_insulin)
        dist_time = Categorical(logits=logits_time)

        action_type = dist_type.sample()
        carb = dist_carb.sample()
        insulin = dist_insulin.sample()
        time = dist_time.sample()

        log_probs = dist_type.log_prob(action_type) + dist_carb.log_prob(carb) + \
                    dist_insulin.log_prob(insulin) + dist_time.log_prob(time)

        return action_type.item(), carb.item(), insulin.item(), time.item(), log_probs

    def get_log_prob(self, state, actions):
        logits_type, logits_carb, logits_insulin, logits_time = self.forward(state)

        dist_type = Categorical(logits=logits_type)
        dist_carb = Categorical(logits=logits_carb)
        dist_insulin = Categorical(logits=logits_insulin)
        dist_time = Categorical(logits=logits_time)

        a_type, a_carb, a_insulin, a_time = actions.T

        log_prob = dist_type.log_prob(a_type) + dist_carb.log_prob(a_carb) + \
                   dist_insulin.log_prob(a_insulin) + dist_time.log_prob(a_time)
        return log_prob


class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, PPOConfig.HIDDEN_SIZE)
        self.fc2 = nn.Linear(PPOConfig.HIDDEN_SIZE, PPOConfig.HIDDEN_SIZE)
        self.value = nn.Linear(PPOConfig.HIDDEN_SIZE, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)


class PPOAgent:
    def __init__(self, state_dim, device, n_carb=50, n_insulin=30, n_time=12):
        self.device = device
        self.actor = PPOActor(state_dim, 3, n_carb, n_insulin, n_time).to(device)
        self.critic = PPOCritic(state_dim).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=PPOConfig.LEARNING_RATE
        )
        self.clip_ratio = PPOConfig.CLIP_RATIO
        self.entropy_coef = PPOConfig.ENTROPY_COEF
        self.value_coef = PPOConfig.VALUE_COEF

        self.total_it = 0  # <--- Counter for training iterations

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor.get_action(state)

    def evaluate(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return self.critic(state).item()

    def train(self, buffer, batch_size=PPOConfig.BATCH_SIZE, epochs=PPOConfig.TRAINING_EPOCHS):
        self.total_it += 1
        states, actions, rewards, old_log_probs, returns, advantages = buffer.get()

        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                s_batch = states[idx:idx+batch_size]
                a_batch = actions[idx:idx+batch_size]
                r_batch = rewards[idx:idx+batch_size]
                logp_old_batch = old_log_probs[idx:idx+batch_size]
                return_batch = returns[idx:idx+batch_size]
                adv_batch = advantages[idx:idx+batch_size]

                logp = self.actor.get_log_prob(s_batch, a_batch)
                ratio = torch.exp(logp - logp_old_batch)

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(s_batch).squeeze()
                critic_loss = F.mse_loss(value, return_batch)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * logp.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Print every 10 training calls, like TD3
        if self.total_it % 10 == 0:
            print(f"[PPO] Step: {self.total_it} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")

