import time
import numpy as np
import psutil
import pgrep

if __name__ == "__main__":
    num_postprocessors = 2
    worker_processes = [psutil.Process(int(pid)) for pid in pgrep.pgrep('Post')]
    assert len(worker_processes) == num_postprocessors
    while True:
        cpu_percents = [p.cpu_percent() for p in worker_processes]
        print("Ray postprocessor CPU average usage: {:.2f}%, maximum usage: {:.2f}%, minimum usage: {:.2f}%".format(
            np.mean(cpu_percents), np.max(cpu_percents), np.min(cpu_percents)))
        time.sleep(1)
