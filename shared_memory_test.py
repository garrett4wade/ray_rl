from multiprocessing.managers import SharedMemoryManager
# from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process, Condition, Lock
# from threading import Thread
# import torch
# import ray
import time
import numpy as np


class StorageProperty:
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


def write1(buffer):
    time.sleep(2)
    while True:
        data = {'main': np.ones((2, np.random.randint(1, 9), 11), dtype=np.float32)}
        buffer.put(data)
        print("write process1 put something")
        time.sleep(1)


def write2(buffer):
    time.sleep(2)
    while True:
        data = {'main': 2 * np.ones((2, np.random.randint(1, 9), 11), dtype=np.float32)}
        buffer.put(data)
        print("write process2 put something")
        time.sleep(1)


def read1(buffer):
    while True:
        data = buffer.get()['obs']
        assert np.all(np.all(data == 1, axis=-1) + np.all(data == 2, axis=-1))
        print("read process1 get something")
        print(data)


def read2(buffer):
    while True:
        data = buffer.get()['obs']
        assert np.all(np.all(data == 1, axis=-1) + np.all(data == 2, axis=-1))
        print("read process2 get something")
        print(data)


if __name__ == "__main__":
    # ray.init()
    sps = {'main': StorageProperty(2, 1, ['obs', 'adv'], [10, 1])}
    buffer = SharedCircularBuffer(maxsize=16, reuse_times=2, storage_properties=sps, produced_batch_size=16)
    p1 = Process(target=write1, args=(buffer, ))
    p2 = Process(target=write2, args=(buffer, ))
    p1.start()
    p2.start()
    pp1 = Process(target=read1, args=(buffer, ))
    pp2 = Process(target=read2, args=(buffer, ))
    pp1.start()
    pp2.start()
    p1.join()
    # time.sleep(60)
    # job = buffer_worker.remote(buffer.data)
    # ray.get(job)
    # print(buffer.data)
