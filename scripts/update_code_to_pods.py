import subprocess
from collections import namedtuple

Pod = namedtuple('Pod', ['name', 'ready', 'status', 'restarts', 'age'])
child1 = subprocess.Popen(["kubectl", 'get', 'pods'], stdout=subprocess.PIPE)
all_pods_str = child1.stdout.read().split()[5:]
pods_num = len(all_pods_str) // 5

pods = []
for i in range(pods_num):
    pods.append(Pod(*all_pods_str[i * 5:(i + 1) * 5]))

for pod in pods:
    subprocess.Popen(['kubectl', 'exec', '-it', pod.name, '--', 'rm', '-rf', '/ray_rl']).wait()
    subprocess.Popen(['kubectl', 'cp', '..', pod.name + ':/ray_rl']).wait()
