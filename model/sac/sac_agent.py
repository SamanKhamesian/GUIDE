import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from config import Action
import numpy as np
from config import SACConfig


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.max_action = torch.FloatTensor(max_action)

        self.fc1 = nn.Linear(state_dim, SACConfig.HIDDEN_SIZE)
        self.fc2 = nn.Linear(SACConfig.HIDDEN_SIZE, SACConfig.HIDDEN_SIZE)

        # Output mean and log_std for ALL action dims
        self.mean = nn.Linear(SACConfig.HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(SACConfig.HIDDEN_SIZE, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = torch.clamp(
            self.log_std(x),
            SACConfig.LOG_STD_MIN,
            SACConfig.LOG_STD_MAX
        )

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)
        raw_action = dist.rsample()

        tanh_action = torch.tanh(raw_action)

        action = tanh_action.clone()

        # 1) action type scores → keep [-1, 1]
        action[:, 0:3] = tanh_action[:, 0:3]

        # 2) carb, insulin, time → map to [0, max]
        action[:, 3:] = (tanh_action[:, 3:] + 1.0) / 2.0 * self.max_action[3:].to(state.device)

        # Correct SAC log-prob (still uses raw_action!)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, SACConfig.HIDDEN_SIZE)
        self.fc2 = nn.Linear(SACConfig.HIDDEN_SIZE, SACConfig.HIDDEN_SIZE)
        self.q = nn.Linear(SACConfig.HIDDEN_SIZE, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device

        self.actor = SACActor(
            state_dim, action_dim, max_action
        ).to(device)

        self.critic1 = SACCritic(state_dim, action_dim).to(device)
        self.critic2 = SACCritic(state_dim, action_dim).to(device)

        self.critic1_target = SACCritic(state_dim, action_dim).to(device)
        self.critic2_target = SACCritic(state_dim, action_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=SACConfig.ACTOR_LR
        )
        self.critic_optim = torch.optim.Adam(
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            lr=SACConfig.CRITIC_LR
        )

        self.alpha = SACConfig.ALPHA
        self.gamma = SACConfig.GAMMA
        self.tau = SACConfig.TAU

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_vec, _ = self.actor.sample(state)

        action_vec = action_vec.cpu().numpy()[0]

        type_scores = action_vec[:3]
        action_type = np.argmax(type_scores)

        carb = action_vec[3]
        insulin = action_vec[4]
        time_index = int(np.clip(round(action_vec[5]), 0, SACConfig.N_TIME_SLOTS - 1))

        if action_type == Action.EAT:
            value = carb
        elif action_type == Action.INJECT:
            value = insulin
        else:
            value = 0.0

        return action_type, value, time_index

    def train(self, replay_buffer, batch_size=SACConfig.BATCH_SIZE):
        self.total_it += 1

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # ----- Critic update -----
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_states)

            q1_t = self.critic1_target(next_states, next_action)
            q2_t = self.critic2_target(next_states, next_action)

            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp.unsqueeze(-1)
            target = rewards + (1 - dones) * self.gamma * q_t

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----- Actor update -----
        action, logp = self.actor.sample(states)

        q1_pi = self.critic1(states, action)
        q2_pi = self.critic2(states, action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp.unsqueeze(-1) - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----- Target update -----
        for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        if self.total_it % 10 == 0:
            print(f"[SAC] Step {self.total_it} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Critic Loss: {critic_loss.item():.4f}")
