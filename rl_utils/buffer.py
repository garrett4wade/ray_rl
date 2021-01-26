import time
import random
import numpy as np
from collections import namedtuple
from torch import from_numpy
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

    def get_storage(self):
        return self._storage


class SharedCircularBuffer:
    def __init__(self, maxsize, chunk_len, reuse_times, shapes, dtypes, num_gpus, bs_per_gpu, verbose_time=False):
        self.reuse_times = reuse_times
        self.maxsize = maxsize
        self.chunk_len = chunk_len
        self.bs_per_gpu = bs_per_gpu
        self.num_read_proc = num_gpus
        self.shapes = shapes
        self.dtypes = dtypes
        self.verbose_time = verbose_time

        self._read_ready = Condition(Lock())

        self._smm = SharedMemoryManager()
        self._smm.start()
        self._storage_shms = []
        storage = []

        for k, shp in shapes.items():
            if '32' in str(dtypes[k]):
                byte_per_digit = 4
            elif 'bool' in str(dtypes[k]) or '8' in str(dtypes[k]):
                byte_per_digit = 1
            else:
                raise NotImplementedError
            if 'rnn_hidden' not in k:
                _storage_shm = self._smm.SharedMemory(size=byte_per_digit * maxsize * chunk_len * np.prod(shp))
                storage.append(np.ndarray((chunk_len, maxsize, *shp), dtype=dtypes[k], buffer=_storage_shm.buf))
            else:
                _storage_shm = self._smm.SharedMemory(size=byte_per_digit * maxsize * np.prod(shp))
                storage.append(np.ndarray((shp[0], maxsize, *shp[1:]), dtype=dtypes[k], buffer=_storage_shm.buf))
            self._storage_shms.append(_storage_shm)
        self.storage = namedtuple('Seg', list(self.shapes.keys()))(*storage)

        self._used_times_shm = self._smm.SharedMemory(size=maxsize)
        self.used_times = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._used_times_shm.buf)

        # if is_free is 1, then the corresponding storage can be written but can't be read
        self._is_free_shm = self._smm.SharedMemory(size=maxsize)
        self.is_free = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._is_free_shm.buf)
        self.is_free[:] = np.ones((maxsize, ), dtype=np.uint8)[:]
        # if is_busy is 1, then the corresponding storage is being written
        self._is_busy_shm = self._smm.SharedMemory(size=maxsize)
        self.is_busy = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._is_busy_shm.buf)
        # is is_ready is 1, then the corresponding storage can be written & read
        self._is_ready_shm = self._smm.SharedMemory(size=maxsize)
        self.is_ready = np.ndarray((maxsize, ), dtype=np.uint8, buffer=self._is_ready_shm.buf)

        self._received_sample_shm = self._smm.SharedMemory(size=8)
        self.received_sample = np.ndarray((), dtype=np.int64, buffer=self._received_sample_shm.buf)

    def size(self):
        return sum(self.is_ready)

    def __len__(self):
        return self.size()

    def put(self, data_batch):
        batch_size = data_batch[0].shape[1]

        t1 = time.time()
        try:
            self._read_ready.acquire()
            tw = time.time()
            indices = np.nonzero(self.is_free)[0][:batch_size]
            self.is_free[indices] = 0
            if len(indices) < batch_size:
                extend_indices = np.nonzero(self.is_ready)[0][:batch_size - len(indices)]
                self.is_ready[extend_indices] = 0
                indices = np.concatenate([indices, extend_indices])
            assert len(indices) == batch_size, ('No enough free & ready indices, try to increase buffer size!')
            self.is_busy[indices] = 1
            # assert np.all(self.is_ready + self.is_busy + self.is_free)
        finally:
            self._read_ready.release()
        t2 = time.time()

        for k in data_batch._fields:
            getattr(self.storage, k)[:, indices] = getattr(data_batch, k).astype(self.dtypes[k])

        t3 = time.time()
        self._read_ready.acquire()
        try:
            self.is_busy[indices] = 0
            self.is_ready[indices] = 1
            self.received_sample += batch_size
            # assert np.all(self.is_ready + self.is_busy + self.is_free)
            if np.sum(self.is_ready) >= self.bs_per_gpu * self.num_read_proc:
                self._read_ready.notify(self.num_read_proc)
        finally:
            self._read_ready.release()
        t4 = time.time()
        if self.verbose_time:
            print(("PUT wait time: {:.2f}ms | " + "preprocess time: {:.2f}ms | " + "copy time: {:.2f}ms | " +
                   "postprocess time: {:.2f}ms").format(1e3 * (tw - t1), 1e3 * (t2 - tw), 1e3 * (t3 - t2),
                                                        1e3 * (t4 - t3)))

    def get(self):
        import time
        t1 = time.time()
        try:
            self._read_ready.acquire()
            self._read_ready.wait_for(lambda: np.sum(self.is_ready) >= self.bs_per_gpu * self.num_read_proc)
            tw = time.time()
            indices = np.random.choice(np.nonzero(self.is_ready)[0], self.bs_per_gpu, replace=False)
            self.is_busy[indices] = 1
            self.is_ready[indices] = 0
        finally:
            self._read_ready.release()
        t2 = time.time()

        data_batch = {}
        for k in self.storage._fields:
            data_batch[k] = from_numpy(getattr(self.storage, k)[:, indices]).cuda()

        t3 = time.time()
        try:
            self._read_ready.acquire()
            self.is_busy[indices] = 0
            self.is_ready[indices] = 1
            # for used-up samples, is_ready -> is_busy (in case of being overwritten) -> is_free
            # for non-used-up samples, is_ready -> is_ready (unchanged)
            self.used_times[indices] += 1
            used_up_indices = indices[self.used_times[indices] >= self.reuse_times]
            self.used_times[used_up_indices] = 0

            self.is_ready[used_up_indices] = 0
            self.is_free[used_up_indices] = 1
        finally:
            # assert np.all(self.is_ready + self.is_busy + self.is_free)
            self._read_ready.release()
        t4 = time.time()
        if self.verbose_time:
            print(("GET wait time: {:.2f}ms | " + "preprocess time: {:.2f}ms | " + "copy time: {:.2f}ms | " +
                   "postprocess time: {:.2f}ms").format(1e3 * (tw - t1), 1e3 * (t2 - tw), 1e3 * (t3 - t2),
                                                        1e3 * (t4 - t3)))
        return data_batch

    def get_util(self):
        return self.size() / self.maxsize

    def get_received_sample(self):
        return self.received_sample.item()


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
