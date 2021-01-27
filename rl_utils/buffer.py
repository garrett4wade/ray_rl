import time
import random
import numpy as np
from collections import namedtuple
from torch import from_numpy
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Lock, Condition


def byte_of_dtype(dtype):
    if '32' in str(dtype):
        byte_per_digit = 4
    elif '64' in str(dtype):
        byte_per_digit = 8
    elif 'bool' in str(dtype) or '8' in str(dtype):
        byte_per_digit = 1
    else:
        raise NotImplementedError
    return byte_per_digit


def shm_array_from_smm(smm, shape, dtype):
    byte_per_digit = byte_of_dtype(dtype)
    shm = smm.SharedMemory(size=byte_per_digit * np.prod(shape))
    shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, shm_array


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
    def __init__(self, q_size, chunk_len, bs_per, world_size, reuse_times, shapes, dtypes, verbose_time=False):
        self.q_size = q_size
        self.chunk_len = chunk_len
        self.bs_per = bs_per
        self.world_size = world_size
        self.reuse_times = reuse_times
        self.shapes = shapes
        self.dtypes = dtypes
        self.verbose_time = verbose_time

        self.num_batches = num_batches = q_size * world_size

        self._read_ready = Condition(Lock())
        self._smm = SharedMemoryManager()
        self._smm.start()

        self._storage_shms = []
        storage = []
        # initialize storage numpy arrays, which are orgnized as a namedtuple
        for k, shp in shapes.items():
            if 'rnn_hidden' in k:
                shm, array = shm_array_from_smm(self._smm, (num_batches, shp[0], bs_per, *shp[1:]), dtypes[k])
            else:
                shm, array = shm_array_from_smm(self._smm, (num_batches, chunk_len, bs_per, *shp), dtypes[k])
            self._storage_shms.append(shm)
            storage.append(array)
        self.storage = namedtuple('Seg', list(self.shapes.keys()))(*storage)

        # indicator of how many times each batch has been used
        self._used_times_shm, self.used_times = shm_array_from_smm(self._smm, (num_batches, ), np.uint8)
        # indicate of whether each batch is ready to be read
        self._is_readable_shm, self.is_readable = shm_array_from_smm(self._smm, (num_batches, ), np.uint8)
        # indicate of whether each batch is ready to be written
        self._is_writable_shm, self.is_writable = shm_array_from_smm(self._smm, (num_batches, ), np.uint8)
        self.is_writable[:] = np.ones((num_batches, ), dtype=np.uint8)[:]

        self._received_sample_shm, self.received_sample = shm_array_from_smm(self._smm, (), np.int64)
        self._glb_wrt_ptr_shm, self.glb_wrt_ptr = shm_array_from_smm(self._smm, (), np.int64)

    def size(self):
        # TODO: it is just count of readable batches
        return sum(self.is_readable)

    def __len__(self):
        return self.size()

    def put(self, seg):
        seg_size = seg[0].shape[1]

        # find available slots to write data, and move global pointer to next position
        t1 = time.time()
        try:
            self._read_ready.acquire()
            tw = time.time()
            # get current batch and slot index
            b_idx, s_idx = (self.glb_wrt_ptr // self.bs_per) % self.num_batches, self.glb_wrt_ptr % self.bs_per
            assert self.is_writable[b_idx] and not self.is_readable[b_idx], (self.is_readable, self.is_writable)
            overflow = s_idx + seg_size >= self.bs_per
            if overflow:
                self.is_writable[b_idx] = 0
                remaining = seg_size + s_idx - self.bs_per
                assert remaining < self.bs_per, 'try to increase batch size'
                all_writable_indices = np.nonzero(self.is_writable)[0]
                nex_b_idx = (b_idx + 1) % self.num_batches
                if len(all_writable_indices) > 0:
                    # first, try to find the next writable batch
                    b_idx2 = nex_b_idx
                    while not self.is_writable[b_idx2]:
                        b_idx2 = (b_idx2 + 1) % self.num_batches
                else:
                    # if there's no writable batch, find the next readable batch and overwrite it
                    all_readable_indices = np.nonzero(self.is_readable)[0]
                    assert len(all_readable_indices) > 0, 'try to increase buffer q_size'
                    b_idx2 = nex_b_idx
                    while not self.is_readable[b_idx2]:
                        b_idx2 = (b_idx2 + 1) % self.num_batches
                    self.is_writable[b_idx2] = 1
                    self.is_readable[b_idx2] = 0
                self.glb_wrt_ptr[()] = b_idx2 * self.bs_per + remaining
            else:
                self.glb_wrt_ptr += seg_size
        finally:
            # for debug
            # assert not np.any(np.logical_and(self.is_readable, self.is_writable))
            self._read_ready.release()
        t2 = time.time()

        if not overflow:
            s = slice(s_idx, s_idx + seg_size)
            for k in seg._fields:
                getattr(self.storage, k)[b_idx, :, s] = getattr(seg, k).astype(self.dtypes[k])
        else:
            cut = self.bs_per - s_idx
            for k in seg._fields:
                getattr(self.storage, k)[b_idx, :, s_idx:] = getattr(seg, k)[:, :cut].astype(self.dtypes[k])
                getattr(self.storage, k)[b_idx2, :, :remaining] = getattr(seg, k)[:, cut:].astype(self.dtypes[k])

        t3 = time.time()
        self._read_ready.acquire()
        try:
            self.received_sample += seg_size
            if overflow:
                self.is_readable[b_idx] = 1
                if np.sum(self.is_readable) >= self.world_size:
                    self._read_ready.notify(self.world_size)
        finally:
            self._read_ready.release()
        t4 = time.time()
        if self.verbose_time:
            print(("PUT wait time: {:.2f}ms | " + "preprocess time: {:.2f}ms | " + "copy time: {:.2f}ms | " +
                   "postprocess time: {:.2f}ms").format(1e3 * (tw - t1), 1e3 * (t2 - tw), 1e3 * (t3 - t2),
                                                        1e3 * (t4 - t3)))

    def get(self, barrier=None):
        import time
        t1 = time.time()
        try:
            self._read_ready.acquire()
            self._read_ready.wait_for(lambda: np.sum(self.is_readable) >= self.world_size)
            tw = time.time()
            b_idx = np.nonzero(self.is_readable)[0][0]
            assert self.is_readable[b_idx] and not self.is_writable[b_idx]
            self.is_readable[b_idx] = 0
        finally:
            self._read_ready.release()
        t2 = time.time()

        if barrier is not None:
            barrier.wait()
        data_batch = {}
        for k in self.storage._fields:
            data_batch[k] = from_numpy(getattr(self.storage, k)[b_idx]).cuda()

        t3 = time.time()
        try:
            self._read_ready.acquire()
            self.used_times[b_idx] += 1
            if self.used_times[b_idx] >= self.reuse_times:
                self.used_times[b_idx] = 0
                self.is_readable[b_idx] = 0
                self.is_writable[b_idx] = 1
            else:
                self.is_readable[b_idx] = 1
        finally:
            # for debug
            # assert not np.any(np.logical_and(self.is_readable, self.is_writable))
            self._read_ready.release()
        t4 = time.time()
        if self.verbose_time:
            print(("GET wait time: {:.2f}ms | " + "preprocess time: {:.2f}ms | " + "copy time: {:.2f}ms | " +
                   "postprocess time: {:.2f}ms").format(1e3 * (tw - t1), 1e3 * (t2 - tw), 1e3 * (t3 - t2),
                                                        1e3 * (t4 - t3)))
        return data_batch

    def get_util(self):
        return self.size() / self.num_batches

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
