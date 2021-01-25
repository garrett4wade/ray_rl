import os
import numpy as np


def find_free_gpu(gpu_num):
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > /dev/shm/gpu_mem_tmp')
    memory_available = [int(x.split()[2]) for x in open('/dev/shm/gpu_mem_tmp', 'r').readlines()]
    return list(np.argsort(memory_available)[-gpu_num:])
