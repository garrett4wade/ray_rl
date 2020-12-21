pkill -9 ray
num_workers=(1)
num_env=1
group_name="sample_speed"
job_name="sample_speed"
num_frames=10000000
seed=678446
for num_worker in ${num_workers[@]}
do
    exp_name=${num_env}"*"${num_worker}"_breakout"
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
                            --q_size 16 \
                            --batch_size 512
    pkill -9 ray
done