import numpy as np
import torch


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
