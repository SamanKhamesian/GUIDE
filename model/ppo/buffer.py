import torch


class PPOBuffer:
    def __init__(self, state_dim, device, gamma=0.99, lam=0.95):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.state_dim = state_dim
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(torch.FloatTensor(state))
        self.actions.append(torch.LongTensor(action))
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.values.append(torch.tensor(value, dtype=torch.float32))
        self.log_probs.append(torch.tensor(log_prob, dtype=torch.float32))
        self.dones.append(torch.tensor(done, dtype=torch.float32))

    def compute_advantages(self):
        states = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        rewards = torch.stack(self.rewards).to(self.device)
        values = torch.stack(self.values).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        dones = torch.stack(self.dones).to(self.device)

        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)

        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, rewards, log_probs, returns, advantages

    def get(self):
        return self.compute_advantages()
