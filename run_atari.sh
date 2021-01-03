pkill -9 ray
num_workers=(10)
num_env=16
group_name="shm_buf"
job_name="shm_buf"
num_frames=10000000
seed=678446
for num_worker in ${num_workers[@]}
do
    exp_name="breakout_"${num_env}"*"${num_worker}
    echo "current experiment ${exp_name}"
    python main_atari.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --num_workers ${num_worker} \
                            --gpu_id 0 \
                            --min_return_chunk_num 16 \
                            --q_size 8 \
                            --batch_size 512
    pkill -9 ray
done