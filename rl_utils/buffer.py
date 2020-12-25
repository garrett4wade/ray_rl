import numpy as np
import random
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Lock, Condition


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

    def put_batch(self, segs):
        self._storage += segs
        self.used_times += [0 for _ in range(len(segs))]
        if len(self._storage) > self._maxsize:
            self._storage = self._storage[-self._maxsize:]
            self.used_times = self.used_times[-self._maxsize:]
        self._next_idx = len(self._storage) % self._maxsize
        self.received_sample += len(segs)

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
        if self.size() < batch_size:
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
        return {k: np.stack([seg[k] for seg in segs], axis=1) for k in self.keys}

    def get_util(self):
        return len(self._storage) / self._maxsize

    def get_received_sample(self):
        return self.received_sample


class CircularBufferNumpy:
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


class StorageProperty:
    """defines basic properties for a specific shared storage block

    Hopefully, all data needed can be concatenated along the last dim, such that
    communication/copy from Ray to main process has the lowest cost.

    However, there are different types of data that can not concatenated together
    e.g. centralized & decentralized data in multiagent environments,
         burn_in data & data for backpropagation
    (or we MAY concatenate them together at the cost of memory)

    Hence we classify all data we need into different storage types, and
    concatenate data of the same type to reduce communication overhead.
    """
    def __init__(self, length, agent_num, keys, simplex_shapes):
        self.length = length
        self.keys = keys
        self.agent_num = agent_num
        self.simplex_shapes = simplex_shapes
        self.split = [sum(self.simplex_shapes[:i]) for i in range(1, len(self.simplex_shapes))]
        self.combined_shape = (sum(self.simplex_shapes), ) if agent_num == 1 else (agent_num, sum(self.simplex_shapes))


class SharedCircularBuffer:
    def __init__(self, maxsize, reuse_times, storage_properties, produced_batch_size):
        self.reuse_times = reuse_times
        self.maxsize = maxsize
        self.produced_batch_size = produced_batch_size
        self.storage_properties = storage_properties
        self.storage_types = list(storage_properties.keys())

        self._read_ready = Condition(Lock())

        self._smm = SharedMemoryManager()
        self._smm.start()

        self._storage_shm = {}
        self.storage = {}
        for storage_type, ppty in storage_properties.items():
            self._storage_shm[storage_type] = self._smm.SharedMemory(size=4 * maxsize * ppty.length *
                                                                     np.prod(ppty.combined_shape))
            self.storage[storage_type] = np.ndarray((ppty.length, maxsize, *ppty.combined_shape),
                                                    dtype=np.float32,
                                                    buffer=self._storage_shm[storage_type].buf)

        self._used_times_shm = self._smm.SharedMemory(size=maxsize)
        self.used_times = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._used_times_shm.buf)

        # if is_free is 1, then this block can be written but can't be read
        self._is_free_shm = self._smm.SharedMemory(size=maxsize)
        self.is_free = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._is_free_shm.buf)
        self.is_free[:] = np.ones((maxsize, ), dtype=np.uint8)[:]
        # if is_busy is 1, then this block is being written
        self._is_busy_shm = self._smm.SharedMemory(size=maxsize)
        self.is_busy = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._is_busy_shm.buf)
        # is is_ready is 1, then this block can be written & read
        self._is_ready_shm = self._smm.SharedMemory(size=maxsize)
        self.is_ready = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._is_ready_shm.buf)

        self.received_sample = 0

    def size(self):
        return sum(self.is_ready)

    def __len__(self):
        return self.size()

    def put(self, data_batch):
        batch_size = data_batch[self.storage_types[0]].shape[1]

        self._read_ready.acquire()
        try:
            indices = np.nonzero(self.is_free)[0][:batch_size]
            self.is_free[indices] = 0
            if len(indices) < batch_size:
                extend_indices = np.nonzero(self.is_ready)[0][:batch_size - len(indices)]
                self.is_ready[extend_indices] = 0
                indices = np.concatenate([indices, extend_indices])
            assert len(indices) == batch_size, 'maxsize - # of busy_indices < batch size! try to increase buffer size!'
            self.is_busy[indices] = 1
            assert np.all(self.is_ready + self.is_busy + self.is_free)
        finally:
            self._read_ready.release()

        for st in self.storage_types:
            self.storage[st][:, indices] = data_batch[st].copy()
        self.used_times[indices] = 0

        self._read_ready.acquire()
        try:
            self.is_busy[indices] = 0
            self.is_ready[indices] = 1
            self.received_sample += batch_size
            assert np.all(self.is_ready + self.is_busy + self.is_free)
            if sum(self.is_ready) >= batch_size:
                self._read_ready.notify_all()
        finally:
            self._read_ready.release()

    def get(self):
        self._read_ready.acquire()
        self._read_ready.wait_for(lambda: sum(self.is_ready) >= self.produced_batch_size)

        indices = np.nonzero(self.is_ready)[0][:self.produced_batch_size]
        data_batch = {st: self.storage[st][:, indices].copy() for st in self.storage_types}

        # for used-up samples, is_ready -> is_free
        self.used_times[indices] += 1
        used_up_indices = indices[self.used_times[indices] >= self.reuse_times]
        self.used_times[used_up_indices] = 0
        self.is_free[used_up_indices] = 1
        self.is_ready[used_up_indices] = 0

        assert np.all(self.is_ready + self.is_busy + self.is_free)
        self._read_ready.release()
        result = {}
        for st in self.storage_types:
            ppty = self.storage_properties[st]
            for k, v in zip(ppty.keys, np.split(data_batch[st], ppty.split, -1)):
                result[k] = v
        return result


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
