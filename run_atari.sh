pkill -9 ray
num_workers=(48)
num_env=16
group_name="scalability"
job_name="scalability"
num_frames=10000000
seed=678446
for num_worker in ${num_workers[@]}
do
    exp_name="qreader_pong_env"${num_env}"worker"${num_worker}
    echo "current experiment ${exp_name}"
    python run_async_ppo.py --exp_name ${exp_name} \
                            --wandb_group ${group_name} \
                            --wandb_job ${job_name} \
                            --total_frames ${num_frames} \
                            --seed ${seed} \
                            --env_num ${num_env} \
                            --num_workers ${num_worker} \
                            --gpu_id 0 \
                            --min_return_chunk_num 16 \
                            --q_size 16 \
                            --batch_size 512 \
                            --no_summary
    pkill -9 ray
done