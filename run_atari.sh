pkill -9 ray
pkill -9 python3.8
rm -rf /dev/shm/*
num_workers=(32)
num_env=16
group_name="shm_buf"
job_name="shm_buf"
num_frames=100000000
seed=678446
for num_worker in ${num_workers[@]}
do
    exp_name="ReadyQueueFullTest_8coll_breakout_"${num_env}"*"${num_worker}
    echo "current experiment ${exp_name}"
    python3.8 main_atari.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --num_workers ${num_worker} \
                            --gpu_id 3 \
                            --min_return_chunk_num 16 \
                            --batch_size 512 \
                            --q_size 16 \
                            --num_writers 4 \
                            --num_supervisors 1 \
                            --num_collectors 8
    pkill -9 ray
    pkill -9 python3.8
    rm -rf /dev/shm/*
done