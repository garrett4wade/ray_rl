import ray
import itertools


@ray.remote(num_returns=2)
def unpack(batch_return):
    data_batch, ep_return = batch_return
    return data_batch, ep_return


@ray.remote
def unroll(*args):
    return list(itertools.chain(*args))


@ray.remote
def isNone(x):
    return x is None
