
import numpy as np
import random

from sum_tree import SumTree


class MemoryDB:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, e, a, beta, beta_increment_per_sampling, capacity, max_priority):
        self.capacity = capacity
        self.e = e
        self.a = a
        self.beta =beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.capacity = capacity
        self.max_priority = max_priority
        self.sum_tree = SumTree(self.capacity)

    def _get_priority(self, error):
        return min((self.max_priority, (error + self.e) ** self.a))

    def add(self, experience, error = None):
        p = self._get_priority(error) if error != None else self.max_priority
        self.sum_tree.add(p, experience)

    def add_batch(self, experiences):
        for experience in experiences:
            self.add(experience, self.max_priority)

    def update(self, index, error, experience):
        p = self._get_priority(error)
        self.sum_tree.update(index, p)

    def update_batch(self, indexes, errors, experiences):
        for index, error, experience in zip(indexes, errors, experiences):
            self.update(index, error, experience)

    def get_experiences_size(self):
        return self.sum_tree.getCount()

    def sample(self, n):
        
        batch = []
        idxs = []
        segment = self.sum_tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.sum_tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.sum_tree.total()
        is_weight = np.power(self.sum_tree.n_entries *
                             sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight
