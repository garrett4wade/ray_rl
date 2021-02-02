import os
import time
import subprocess
import argparse
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('--copy', action='store_true', help='copy code to pods')
parser.add_argument('--restart', action='store_true', help='restart pods')
args = parser.parse_args()

if __name__ == "__main__":
    if args.restart:
        print("restarting pods ...")
        os.system('kubectl delete -f ~/workspace/gpupool/distributed/ray.yaml')
        os.system('kubectl apply -f ~/workspace/gpupool/distributed/ray.yaml')
        print("restart finished, waiting for pods ready ...")
        time.sleep(30)

    Pod = namedtuple('Pod', ['name', 'ready', 'status', 'restarts', 'age'])
    child1 = subprocess.Popen(["kubectl", 'get', 'pods'], stdout=subprocess.PIPE)
    all_pods_str = [s.decode() for s in child1.stdout.read().split()[5:]]
    pods_num = len(all_pods_str) // 5

    pods = []
    for i in range(pods_num):
        pods.append(Pod(*all_pods_str[i * 5:(i + 1) * 5]))

    if args.restart:
        wandb_apikey = '6bda9cf410e5e5d857441c4d0b3a1a8cb82b4ea3'
    for pod in pods:
        if pod.status == 'Running' and 'head' in pod.name:
            os.system('kubectl exec -it {} -- wandb login '.format(pod.name) + wandb_apikey)

    if args.copy:
        # copy codes to pods
        code_folder = '~/workspace/ray_rl'
        destination = '/ray_rl'
        for pod in pods:
            if pod.status == 'Running':
                os.system('kubectl exec -it {} -- rm -rf '.format(pod.name) + destination)
                os.system('kubectl cp ' + code_folder + ' {}:'.format(pod.name) + destination)
                print('finished copy onto {}.'.format(pod.name))
            else:
                print("pod {} is not running yet, ignore copy onto this pod.".format(pod.name))

    if not any(vars(args).values()):
        print("Nothing happens to your {} pods.".format(pods_num))
