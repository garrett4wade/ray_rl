import threading
import numpy as np


class EpisodeRecorder(threading.Thread):
    def __init__(self, info_queue):
        super().__init__()
        self.info_queue = info_queue
        self.storage = dict()
        self.statistics = dict()

    def run(self):
        while True:
            infos = self.info_queue.get()
            info_keys = infos[0]._fields
            info_dict = {k: [getattr(info, k) for info in infos] for k in info_keys}
            for k, v in info_dict.items():
                if k not in self.storage.keys():
                    self.storage[k] = v
                else:
                    self.storage[k] += v

    def pull(self):
        if len(self.storage) > 0:
            for k, v in self.storage.items():
                self.statistics[k + '/max'] = np.max(v)
                self.statistics[k + '/min'] = np.min(v)
                self.statistics[k + '/avg'] = np.mean(v)
            self.storage = dict()
        return self.statistics
