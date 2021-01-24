import subprocess
import argparse
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('--copy', action='store_true', help='copy code to pods')
args = parser.parse_args()

if __name__ == "__main__":
    Pod = namedtuple('Pod', ['name', 'ready', 'status', 'restarts', 'age'])
    child1 = subprocess.Popen(["kubectl", 'get', 'pods'], stdout=subprocess.PIPE)
    all_pods_str = [s.decode() for s in child1.stdout.read().split()[5:]]
    pods_num = len(all_pods_str) // 5

    pods = []
    for i in range(pods_num):
        pods.append(Pod(*all_pods_str[i * 5:(i + 1) * 5]))

    if args.copy:
        # copy codes to pods
        for pod in pods:
            subprocess.Popen(['kubectl', 'exec', '-it', pod.name, '--', 'rm', '-rf', '/ray_rl']).wait()
            subprocess.Popen(['kubectl', 'cp', '..', pod.name + ':/ray_rl']).wait()
            print('finishing copy onto {}'.format(pod))
