import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from config import PPOConfig


class PPOActor(nn.Module):
    def __init__(self, state_dim, n_action_type=3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, PPOConfig.HIDDEN_SIZE)
        self.fc2 = nn.Linear(PPOConfig.HIDDEN_SIZE, PPOConfig.HIDDEN_SIZE)

        # Output means and log_stds for carb, insulin, time
        self.mean_carb = nn.Linear(PPOConfig.HIDDEN_SIZE, 1)
        self.mean_insulin = nn.Linear(PPOConfig.HIDDEN_SIZE, 1)
        self.mean_time = nn.Linear(PPOConfig.HIDDEN_SIZE, 1)

        self.log_std_carb = nn.Parameter(torch.zeros(1))
        self.log_std_insulin = nn.Parameter(torch.zeros(1))
        self.log_std_time = nn.Parameter(torch.zeros(1))

        # Still use categorical for action_type (EAT/INJECT/NOTHING)
        self.logits_type = nn.Linear(PPOConfig.HIDDEN_SIZE, n_action_type)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean_carb = self.mean_carb(x)
        mean_insulin = self.mean_insulin(x)
        mean_time = self.mean_time(x)
        logits_type = self.logits_type(x)
        return mean_carb, mean_insulin, mean_time, logits_type

    def get_action(self, state):
        mean_carb, mean_insulin, mean_time, logits_type = self.forward(state)

        std_carb = self.log_std_carb.exp().clamp(min=1e-3, max=1e2)
        std_insulin = self.log_std_insulin.exp().clamp(min=1e-3, max=1e2)
        std_time = self.log_std_time.exp().clamp(min=1e-3, max=1e2)

        mean_carb = mean_carb.clamp(PPOConfig.CARB_RANGE[0], PPOConfig.CARB_RANGE[1])
        mean_insulin = mean_insulin.clamp(PPOConfig.INSULIN_RANGE[0], PPOConfig.INSULIN_RANGE[1])
        mean_time = mean_time.clamp(PPOConfig.TIME_INDEX_RANGE[0], PPOConfig.TIME_INDEX_RANGE[1])

        dist_carb = Normal(mean_carb, std_carb)
        dist_insulin = Normal(mean_insulin, std_insulin)
        dist_time = Normal(mean_time, std_time)
        dist_type = torch.distributions.Categorical(logits=logits_type)

        # Sample actions
        carb = dist_carb.sample().squeeze()
        insulin = dist_insulin.sample().squeeze()
        time = dist_time.sample().squeeze()
        action_type = dist_type.sample().item()

        # Torch clamp to config bounds
        carb = torch.clamp(carb, PPOConfig.CARB_RANGE[0], PPOConfig.CARB_RANGE[1]).item()
        insulin = torch.clamp(insulin, PPOConfig.INSULIN_RANGE[0], PPOConfig.INSULIN_RANGE[1]).item()
        time = int(torch.clamp(time, PPOConfig.TIME_INDEX_RANGE[0], PPOConfig.TIME_INDEX_RANGE[1]).item())

        # Log probability: always use the tensor values (not Python floats)
        carb_tensor = torch.tensor(carb).to(mean_carb.device)
        insulin_tensor = torch.tensor(insulin).to(mean_insulin.device)
        time_tensor = torch.tensor(time).to(mean_time.device)
        action_type_tensor = torch.tensor(action_type).to(logits_type.device)

        log_prob = (dist_carb.log_prob(carb_tensor) +
                    dist_insulin.log_prob(insulin_tensor) +
                    dist_time.log_prob(time_tensor) +
                    dist_type.log_prob(action_type_tensor))

        return action_type, carb, insulin, time, log_prob

    def get_log_prob(self, state, actions):
        # actions: [action_type, carb, insulin, time]
        mean_carb, mean_insulin, mean_time, logits_type = self.forward(state)
        std_carb = self.log_std_carb.exp()
        std_insulin = self.log_std_insulin.exp()
        std_time = self.log_std_time.exp()

        dist_carb = Normal(mean_carb, std_carb)
        dist_insulin = Normal(mean_insulin, std_insulin)
        dist_time = Normal(mean_time, std_time)
        dist_type = torch.distributions.Categorical(logits=logits_type)

        a_type, a_carb, a_insulin, a_time = actions.T

        log_prob = (
                dist_type.log_prob(a_type) + dist_carb.log_prob(a_carb.unsqueeze(-1)) + dist_insulin.log_prob(a_insulin.unsqueeze(-1)) + dist_time.log_prob(
            a_time.unsqueeze(-1)))
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
    def __init__(self, state_dim, device):
        self.device = device
        self.actor = PPOActor(state_dim, 3).to(device)
        self.critic = PPOCritic(state_dim).to(device)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=PPOConfig.LEARNING_RATE)
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

    def train(self, buffer, batch_size=PPOConfig.BATCH_SIZE, epochs=PPOConfig.MAX_EPOCHS):
        self.total_it += 1
        states, actions, rewards, old_log_probs, returns, advantages = buffer.get()

        for _ in range(epochs):
            for idx in range(0, len(states), batch_size):
                s_batch = states[idx:idx + batch_size]
                a_batch = actions[idx:idx + batch_size]
                r_batch = rewards[idx:idx + batch_size]
                logp_old_batch = old_log_probs[idx:idx + batch_size]
                return_batch = returns[idx:idx + batch_size]
                adv_batch = advantages[idx:idx + batch_size]

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
