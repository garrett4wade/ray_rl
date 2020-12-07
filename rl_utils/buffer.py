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
            k:
            np.stack([seg[k]
                      for seg in segs], axis=0) if k != 'hidden_state' else np.stack([seg[k] for seg in segs], axis=1)
            for k in self.keys
        }


class ReplayQueue():
    def __init__(self, maxsize, keys):
        self._storage = []
        self._maxsize = int(maxsize)
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
        self._next_idx = len(self._storage)
        return {k: np.stack([seg[k] for seg in segs], axis=0) for k in self.keys}


class CircularBuffer:
    def __init__(self, maxsize, reuse_times, keys):
        self._storage = []

        self._maxsize = int(maxsize)
        assert isinstance(reuse_times, int) and reuse_times >= 1
        self.reuse_times = reuse_times
        self.keys = keys
        self._next_idx = 0

        self.used_times = []
        self.received_sample = 0

    def size(self):
        return len(self._storage)

    def __len__(self):
        return len(self._storage)

    def put(self, seg):
        if self._next_idx >= len(self._storage):
            self._storage.append(seg)
            self.used_times.append(0)
        else:
            self._storage[self._next_idx] = seg
            self.used_times[self._next_idx] = 0
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self.received_sample += 1

    def get(self, batch_size):
        if self.size() <= batch_size:
            return None
        idxes = np.random.choice(self.size(), batch_size, replace=False)
        idxes = sorted(idxes, reverse=True)
        segs = []
        for i in idxes:
            if self.used_times[i] >= self.reuse_times - 1:
                segs.append(self._storage.pop(i))
                self.used_times.pop(i)
            else:
                segs.append(self._storage[i])
                self.used_times[i] += 1
        self._next_idx = len(self._storage)
        return {k: np.stack([seg[k] for seg in segs], axis=0) for k in self.keys}


class CircularBuffer2:
    def __init__(self, maxsize, reuse_times, keys, shapes, pad_values):
        self._storage = {}
        for k in keys:
            init_value = np.zeros if pad_values[k] == 0 else np.ones
            self._storage[k] = init_value((maxsize, *shapes[k]), dtype=np.float32)

        self._maxsize = int(maxsize)
        self.keys = keys

        self.reuse_times = reuse_times
        self.used_times = np.zeros((maxsize, ), dtype=np.int32)

        self.free_indices = list(range(maxsize))
        self.full_indices = []
        self.received_sample = 0

    def size(self):
        return len(self.full_indices)

    def __len__(self):
        return self.size()

    def put(self, data_batch):
        batch_size = len(data_batch[self.keys[0]])
        indices = self.free_indices[:batch_size]
        if len(indices) < batch_size:
            indices.extend(self.full_indices[:batch_size - len(indices)])

        for k, v in data_batch.items():
            self._storage[k][indices] = v

        self.full_indices.extend(self.free_indices[:batch_size])
        self.free_indices = self.free_indices[batch_size:]
        self.used_times[indices] = 0
        self.received_sample += batch_size

    def get(self, batch_size):
        if len(self.full_indices) < batch_size:
            return None
        indices = np.random.choice(self.full_indices, batch_size, replace=False)
        data_batch = {}
        for k, v in self._storage.items():
            data_batch[k] = v[indices]

        self.used_times[indices] += 1
        used_up_indices = indices[self.used_times[indices] >= self.reuse_times]
        self.used_times[used_up_indices] = 0
        self.free_indices.extend(used_up_indices)
        for idx in used_up_indices:
            self.full_indices.remove(idx)
        return data_batch


class ReplayBuffer():
    def __init__(self, maxsize, keys, shapes, pad_values):
        self._storage = {}
        for k in keys:
            init_value = np.zeros if pad_values[k] == 0 else np.ones
            self._storage[k] = init_value((maxsize, *shapes[k]), dtype=np.float32)

        self._maxsize = int(maxsize)
        self._filled = self._next_idx = 0
        self.keys = keys

    def size(self):
        return self._filled

    def __len__(self):
        return self._filled

    def put(self, data_batch):
        batch_size = len(data_batch[self.keys[0]])
        if self._next_idx + batch_size > self._maxsize:
            remaining = self._next_idx + batch_size - self._maxsize
            idx = np.concatenate([np.arange(self._next_idx, self._maxsize), np.arange(remaining)], axis=0)
        else:
            idx = np.arange(self._next_idx, self._next_idx + batch_size)

        for k, v in data_batch.items():
            self._storage[k][idx] = v

        self._next_idx = (self._next_idx + batch_size) % self._maxsize
        self._filled = min(self._filled + batch_size, self._maxsize)

    def get(self, batch_size):
        if self.size() <= batch_size:
            return None
        idx = np.random.choice(self.size(), batch_size, replace=False)
        data_batch = {}
        for k, v in self._storage.items():
            data_batch[k] = v[idx]
        return data_batch


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
        idxes = np.random.choice(self.size(), batch_size, replace=False, p=self._prioritization / total_priorities)
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
