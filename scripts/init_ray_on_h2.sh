ssh fuwei@192.168.1.102 -i ~/.ssh/id_rsa & ray start --head --node-ip-address=192.168.1.102 --redis-shard-ports=6380,6381 --port=6379 --num-cpus=60 --object-manager-port=12345 --node-manager-port=12346 --object-store-memory=60000000000 --resources='{"head":1000}'
pip install -U ray
ray start --node-ip-address=192.168.1.103 \
          --num-cpus=60 \
          --object-manager-port=12345 \
          --node-manager-port=12346 \
          --address=192.168.1.102:6379
