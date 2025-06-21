import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor network maps state to continuous action output ∈ [0, max_action]
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
            # Ensure output in [-1, 1], scaled later by max_action
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


# Critic network estimates Q-values for (state, action) pairs
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


# TD3 with Behavior Cloning loss (TD3-BC)
class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 2.5  # scaling for BC loss term
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    # Select action from actor (inference mode)
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    # One training step using batch from replay buffer
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size, device=self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.gamma * torch.min(target_Q1, target_Q2)

        # Update critic networks
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % 1000 == 0:
            print(f"[TD3-BC] Step: {self.total_it} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")

