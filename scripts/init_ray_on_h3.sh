pip install -U ray
ray start --head \
          --node-ip-address=192.168.1.103 \
          --port=6379 \
          --redis-shard-ports=6380,6381 \
          --num-cpus=60 \
          --object-manager-port=12345 \
          --node-manager-port=12346 \
          --object-store-memory=60000000000 \
          --resources='{"head":1000}'
ssh fuwei@192.168.1.102 -i ~/.ssh/id_rsa & ray start --node-ip-address=192.168.1.102 --num-cpus=60 --address=192.168.1.103:6379 --object-manager-port=12345 --node-manager-port=12346
