import multiprocessing as mp
from torch import ones as th_ones


class BufferCollector(mp.Process):
    def __init__(self, collector_id, buffer, shm_tensor_dict, available_flag, sample_ready):
        super().__init__()
        self.daemon = True
        self.id = collector_id
        self.buffer = buffer
        self.shm_tensor_dict = shm_tensor_dict
        self.available_flag = available_flag
        self.sample_ready = sample_ready

    def run(self):
        while True:
            try:
                self.sample_ready.acquire()
                self.sample_ready.wait_for(lambda: self.available_flag == 0)
                self.buffer.get(self.shm_tensor_dict)
                self.available_flag.copy_(th_ones(1))
            finally:
                self.sample_ready.release()
