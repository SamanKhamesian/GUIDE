import copy
import torch
import torch.nn.functional as F

from config import CQLConfig
from model.td3_bc.td3_bc_agent import Actor, Critic


class CQL:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device

        # =======================
        # Networks
        # =======================
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=CQLConfig.ACTOR_LR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CQLConfig.CRITIC_LR)

        # =======================
        # Hyperparameters
        # =======================
        self.gamma = CQLConfig.GAMMA
        self.tau = CQLConfig.TAU
        self.alpha = CQLConfig.CQL_ALPHA
        self.policy_freq = CQLConfig.POLICY_FREQ

        self.total_it = 0

        # =======================
        # Action bounds
        # =======================
        self.action_high = torch.tensor(max_action, device=device)
        self.action_low = torch.zeros_like(self.action_high)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=CQLConfig.BATCH_SIZE):
        self.total_it += 1

        # =======================
        # Sample batch
        # =======================
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size, to_tensor=True, device=self.device)

        # =======================
        # Bellman target (TD3-style)
        # =======================
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.gamma * torch.min(target_Q1, target_Q2)

        # =======================
        # Critic loss (Bellman)
        # =======================
        current_Q1, current_Q2 = self.critic(state, action)
        bellman_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

        # =======================
        # CQL loss
        # =======================
        B, A = action.shape
        N = CQLConfig.NUM_RANDOM_ACTIONS

        random_actions = (self.action_low + (self.action_high - self.action_low) * torch.rand(B * N, A, device=self.device))

        state_repeat = state.unsqueeze(1).repeat(1, N, 1).view(-1, state.shape[1])

        q1_rand, q2_rand = self.critic(state_repeat, random_actions)
        q1_rand = q1_rand.view(B, N)
        q2_rand = q2_rand.view(B, N)

        cql_q1 = torch.logsumexp(q1_rand, dim=1).mean() - current_Q1.mean()
        cql_q2 = torch.logsumexp(q2_rand, dim=1).mean() - current_Q2.mean()

        cql_loss = self.alpha * (cql_q1 + cql_q2)

        critic_loss = bellman_loss + cql_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # =======================
        # Actor update (CQL-BC)
        # =======================
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)

            # Q maximization term
            q_loss = -self.critic.Q1(state, pi).mean()

            # Behavioral cloning term (anchors to dataset)
            bc_loss = F.mse_loss(pi, action)

            # Final CQL-BC actor loss
            actor_loss = q_loss + CQLConfig.BC_WEIGHT * bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # =======================
            # Target network updates
            # =======================
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            # Optional but VERY useful debug print
            if self.total_it % 1000 == 0:
                print(
                    f"[Actor] step {self.total_it} | "
                    f"Q: {q_loss.item():.3f} | "
                    f"BC: {bc_loss.item():.3f}"
                )

        # =======================
        # Logging
        # =======================
        if self.total_it % 1000 == 0:
            print(
                f"[CQL] Step {self.total_it} | "
                f"Bellman: {bellman_loss.item():.3f} | "
                f"CQL: {cql_loss.item():.3f}"
            )
