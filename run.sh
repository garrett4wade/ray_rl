#!/bin/bash
for bs in 1024 512 256 128 64
do
exp_name="async_batch${bs}"
echo ${exp_name}
python run_async_ppo.py --seed 257689 \
                        --exp_name ${exp_name} \
                        --num_workers 12 \
                        --batch_size ${bs} \
                        --total_frames 10000000 \
                        --total_steps 5000
done