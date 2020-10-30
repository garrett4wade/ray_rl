#!/bin/bash
for bs in 512
do
exp_name1="async_ppg_test0"
echo ${exp_name1}
python run_async_ppg.py --seed 257689 \
                        --exp_name ${exp_name1} \
                        --num_workers 12 \
                        --batch_size ${bs} \
                        --total_frames 10000000 \
                        --total_steps 2000
exp_name2="sync_ppg_test0"
echo ${exp_name2}
python run_sync_ppg.py  --seed 257689 \
                        --exp_name ${exp_name2} \
                        --num_workers 12 \
                        --batch_size ${bs} \
                        --total_frames 10000000 \
                        --total_steps 2000
done