import numpy as np
import random


class FIFOQueue():
    def __init__(self, size, keys):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.keys = keys

    def size(self):
        return len(self._storage)

    def __len__(self):
        return len(self._storage)

    def put(self, seg):
        if self._next_idx >= len(self._storage):
            self._storage.append(seg)
        else:
            self._storage[self._next_idx] = seg
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def get(self, batch_size):
        if self.size() <= batch_size:
            return None
        segs = self._storage[:batch_size]
        self._storage = self._storage[batch_size:]
        # random.shuffle(segs)

        return {
            k: np.stack([seg[k] for seg in segs], axis=0) if
            k != 'hidden_state' else np.stack([seg[k] for seg in segs], axis=1)
            for k in self.keys
        }


class ReplayQueue():
    def __init__(self, size, keys):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.keys = keys

    def size(self):
        return len(self._storage)

    def __len__(self):
        return len(self._storage)

    def put(self, seg):
        if self._next_idx >= len(self._storage):
            self._storage.append(seg)
        else:
            self._storage[self._next_idx] = seg
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def get(self, batch_size):
        if self.size() <= batch_size:
            return None
        idxes = np.random.choice(self.size(), batch_size, replace=False)
        idxes = sorted(idxes, reverse=True)
        segs = [self._storage.pop(i) for i in idxes]
        random.shuffle(segs)

        return {
            k: np.stack([seg[k] for seg in segs], axis=0) if
            k != 'hidden_state' else np.stack([seg[k] for seg in segs], axis=1)
            for k in self.keys
        }


class ReplayBuffer():
    def __init__(self, size, keys):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.keys = keys

    def size(self):
        return len(self._storage)

    def __len__(self):
        return len(self._storage)

    def put(self, seg):
        if self._next_idx >= len(self._storage):
            self._storage.append(seg)
        else:
            self._storage[self._next_idx] = seg
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def get(self, batch_size):
        if self.size() <= batch_size:
            return None
        idxes = np.random.choice(self.size(), batch_size, replace=False)
        segs = [self._storage[i] for i in idxes]

        return {
            k: np.stack([seg[k] for seg in segs], axis=0) if
            k != 'hidden_state' else np.stack([seg[k] for seg in segs], axis=1)
            for k in self.keys
        }


class PrioritizedReplayBuffer():
    def __init__(self, size):
        self._storage = []
        self._prioritization = []
        self._maxsize = int(size)
        self._next_idx = 0

    def size(self):
        return len(self._storage)

    def __len__(self):
        return len(self._storage)

    def put(self, seg, p):
        if self._next_idx >= len(self._storage):
            self._storage.append(seg)
            self._prioritization.append(p)
        else:
            self._storage[self._next_idx] = seg
            self._prioritization[self._next_idx] = p
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def add_batch(self, batch, p):
        batch_size = p.shape[0]
        for i in range(batch_size):
            seg = dict(obs=batch['obs'][i],
                       act=batch['act'][i],
                       nex_obs=batch['nex_obs'][i],
                       r=batch['r'][i],
                       d=batch['d'][i])
            self.add(seg, p[i])

    def get(self, batch_size):
        if self.size() <= batch_size:
            return None
        total_priorities = np.sum(self._prioritization)
        idxes = np.random.choice(self.size(),
                                 batch_size,
                                 replace=False,
                                 p=self._prioritization / total_priorities)
        idxes = sorted(idxes, reverse=True)
        segs = [self._storage.pop(i) for i in idxes]
        random.shuffle(segs)

        priorities = [self._prioritization.pop(i) for i in idxes]
        return dict(obs=np.stack([seg['obs'] for seg in segs], axis=0),
                    act=np.stack([seg['act'] for seg in segs], axis=0),
                    nex_obs=np.stack([seg['nex_obs'] for seg in segs], axis=0),
                    r=np.stack([seg['r'] for seg in segs], axis=0),
                    d=np.stack([seg['d'] for seg in segs], axis=0),
                    prob=np.stack(priorities, axis=0) / total_priorities)
